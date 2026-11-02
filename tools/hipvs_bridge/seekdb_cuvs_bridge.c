/* Platform-safe bridge: cuVS brute_force exact kNN graph (k=256) ->
 * CAGRA index with compact 128-degree graph; full 256 graph and FP32 base
 * persisted as an embedded sidecar for host-side expansion/rerank.
 * No custom HIP kernels, no pinned memory, no device-graph import. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct { float d; unsigned id; } Ranked;
static int ranked_less(const void* a, const void* b){
  const Ranked* x=(const Ranked*)a; const Ranked* y=(const Ranked*)b;
  if(x->d<y->d) return -1; if(x->d>y->d) return 1;
  return x->id<y->id?-1:(x->id>y->id?1:0);
}
typedef struct {
  cuvsResources_t res;
  cuvsCagraIndex_t idx;
  long n, dim, k;          /* k = full graph degree */
  long index_k;            /* compact CAGRA degree */
  float* host_base;
  uint32_t* host_graph;
  uint32_t* stamp;
  uint32_t stamp_gen;
} sk_bf_index;

static int env_int(const char* key,int dflt){const char* v=getenv(key);return v&&*v?atoi(v):dflt;}
static void set_tensor(DLManagedTensor* t,void* data,DLDeviceType dev,int bits,long r,long c,long* shape){
  shape[0]=r; shape[1]=c; t->dl_tensor.data=data; t->dl_tensor.device.device_type=dev;
  t->dl_tensor.device.device_id=0; t->dl_tensor.ndim=2; t->dl_tensor.dtype.code=kDLFloat;
  t->dl_tensor.dtype.bits=bits; t->dl_tensor.dtype.lanes=1; t->dl_tensor.shape=shape;
  t->dl_tensor.strides=0; t->dl_tensor.byte_offset=0;
}
static float l2(const float* a,const float* b,long d){float s=0.f;for(long i=0;i<d;i++){float z=a[i]-b[i];s=fmaf(z,z,s);}return s;}

void* seekdb_cuvs_build(const float* base,long n,long dim)
{
  sk_bf_index* h=(sk_bf_index*)calloc(1,sizeof(*h));
  if(!h) return NULL;
  h->n=n; h->dim=dim; h->k=env_int("BF_K",256);
  h->index_k=env_int("BF_INDEX_K",128);
  if(h->index_k<=0||h->index_k>h->k) h->index_k=h->k;

  cuvsBruteForceIndex_t bf=NULL; DLManagedTensor dset; long ds[2];
  long chunk=0; int64_t* h_ids=NULL; uint32_t* h_graph=NULL; int64_t* d_ids=NULL; float* d_dist=NULL;
  uint32_t* g_compact=NULL; cuvsFilter filter; filter.type=NO_FILTER; filter.addr=(uintptr_t)0; int ok=1;

  if(cuvsResourcesCreate(&h->res)!=CUVS_SUCCESS){free(h);return NULL;}
  void* d_dataset=NULL;
  if(cudaMalloc(&d_dataset,(size_t)n*dim*4)!=cudaSuccess) goto fail;
  if(cudaMemcpy(d_dataset,base,(size_t)n*dim*4,cudaMemcpyHostToDevice)!=cudaSuccess) goto fail;
  if(cuvsBruteForceIndexCreate(&bf)!=CUVS_SUCCESS) goto fail;
  set_tensor(&dset,d_dataset,kDLCUDA,32,n,dim,ds);
  if(cuvsBruteForceBuild(h->res,&dset,L2Expanded,0.f,bf)!=CUVS_SUCCESS){fprintf(stderr,"[bf256] bf build: %s\n",cuvsGetLastErrorText());cuvsBruteForceIndexDestroy(bf);goto fail;}
  chunk=env_int("BF_CHUNK",8192); if(chunk<=0)chunk=32768; if(chunk>n)chunk=n;
  if(cudaMalloc((void**)&d_ids,(size_t)chunk*h->k*8)!=cudaSuccess){cuvsBruteForceIndexDestroy(bf);goto fail;}
  if(cudaMalloc((void**)&d_dist,(size_t)chunk*h->k*4)!=cudaSuccess){cuvsBruteForceIndexDestroy(bf);goto fail;}
  h_ids=(int64_t*)malloc((size_t)chunk*h->k*8);
  h_graph=(uint32_t*)malloc((size_t)n*h->k*4);
  if(!h_ids||!h_graph){cuvsBruteForceIndexDestroy(bf);goto fail;}
  for(long start=0;start<n;start+=chunk){
    long m=n-start; if(m>chunk)m=chunk;
    DLManagedTensor qt=dset; long qs[2]={m,dim};
    qt.dl_tensor.data=(char*)d_dataset+(size_t)start*dim*4; qt.dl_tensor.shape=qs;
    DLManagedTensor nb,db; long os[2]={m,h->k};
    set_tensor(&nb,d_ids,kDLCUDA,64,m,h->k,os); nb.dl_tensor.dtype.code=kDLInt;
    set_tensor(&db,d_dist,kDLCUDA,32,m,h->k,os);
    if(cuvsBruteForceSearch(h->res,bf,&qt,&nb,&db,filter)!=CUVS_SUCCESS){fprintf(stderr,"[bf256] bf chunk %ld: %s\n",start,cuvsGetLastErrorText());ok=0;break;}
    if(cudaMemcpy(h_ids,d_ids,(size_t)m*h->k*8,cudaMemcpyDeviceToHost)!=cudaSuccess){ok=0;break;}
    size_t off=(size_t)start*h->k;
    for(long i=0;i<m*h->k;i++) h_graph[off+i]=(uint32_t)h_ids[i];
  }
  cuvsBruteForceIndexDestroy(bf); bf=NULL;
  cudaFree(d_ids); d_ids=NULL; cudaFree(d_dist); d_dist=NULL; free(h_ids); h_ids=NULL;
  if(!ok) goto fail;
  cudaFree(d_dataset); d_dataset=NULL;

  if(cuvsCagraIndexCreate(&h->idx)!=CUVS_SUCCESS) goto fail;
  uint32_t* g_index=h_graph;
  if(h->index_k<h->k){
    g_compact=(uint32_t*)malloc((size_t)n*h->index_k*4);
    if(!g_compact) goto fail;
    for(long row=0;row<n;row++) memcpy(g_compact+(size_t)row*h->index_k,h_graph+(size_t)row*h->k,(size_t)h->index_k*4);
    g_index=g_compact;
  }
  DLManagedTensor graph; long gs[2];
  set_tensor(&graph,g_index,kDLCPU,32,n,h->index_k,gs); graph.dl_tensor.dtype.code=kDLUInt;
  DLManagedTensor hset; long hs[2];
  set_tensor(&hset,(void*)base,kDLCPU,32,n,dim,hs);
  if(cuvsCagraIndexFromArgs(h->res,L2Expanded,&graph,&hset,h->idx)!=CUVS_SUCCESS){fprintf(stderr,"[bf256] cagra from args: %s\n",cuvsGetLastErrorText());goto fail;}
  free(g_compact); g_compact=NULL;

  h->host_base=(float*)malloc((size_t)n*dim*4);
  if(!h->host_base) goto fail;
  memcpy(h->host_base,base,(size_t)n*dim*4);
  h->host_graph=h_graph; h_graph=NULL;
  fprintf(stderr,"[bf256] built n=%ld dim=%ld k=%ld index_k=%ld\n",n,dim,h->k,h->index_k);
  return h;
fail:
  if(bf) cuvsBruteForceIndexDestroy(bf);
  if(d_ids) cudaFree(d_ids); if(d_dist) cudaFree(d_dist); free(h_ids); free(h_graph); free(g_compact);
  if(d_dataset) cudaFree(d_dataset);
  if(h->idx) cuvsCagraIndexDestroy(h->idx);
  if(h->res) cuvsResourcesDestroy(h->res);
  free(h->host_base); free(h->host_graph); free(h);
  return NULL;
}

int seekdb_cuvs_search(void* handle,const float* query,long nq,long topk,unsigned* out_ids,float* out_dist)
{
  sk_bf_index* h=(sk_bf_index*)handle; if(!h) return 1;
  int rc=0; float* dq=NULL; unsigned* dids=NULL; float* ddist=NULL; cuvsCagraSearchParams_t sp=NULL;
  long search_k=topk;
  int seeds=env_int("BF_EXPAND_SEEDS",0), second=env_int("BF_SECOND_HOP",0), deg=env_int("BF_EXPAND_DEGREE",0);
  if(deg<=0||deg>h->k) deg=(int)h->k;
  unsigned* hids=NULL; Ranked* ranked=NULL; size_t cap=(size_t)search_k;
  if(h->host_graph&&h->host_base&&seeds>0) cap+=(size_t)(seeds+(second>0?second:0))*deg;
  if(cudaMalloc((void**)&dq,(size_t)nq*h->dim*4)!=cudaSuccess){rc=10;goto done;}
  if(cudaMalloc((void**)&dids,(size_t)nq*search_k*4)!=cudaSuccess){rc=11;goto done;}
  if(cudaMalloc((void**)&ddist,(size_t)nq*search_k*4)!=cudaSuccess){rc=12;goto done;}
  if(cudaMemcpy(dq,query,(size_t)nq*h->dim*4,cudaMemcpyHostToDevice)!=cudaSuccess){rc=13;goto done;}
  {
    DLManagedTensor qt,nt,dt; long qs[2]={nq,h->dim},os[2]={nq,search_k};
    set_tensor(&qt,dq,kDLCUDA,32,nq,h->dim,qs);
    set_tensor(&nt,dids,kDLCUDA,32,nq,search_k,os); nt.dl_tensor.dtype.code=kDLUInt;
    set_tensor(&dt,ddist,kDLCUDA,32,nq,search_k,os);
    if(cuvsCagraSearchParamsCreate(&sp)!=CUVS_SUCCESS){rc=14;goto done;}
    if(sp->itopk_size<(size_t)env_int("BF_ITOPK",512)) sp->itopk_size=env_int("BF_ITOPK",512);
    sp->search_width=env_int("BF_WIDTH",32);
    cuvsFilter filter; filter.type=NO_FILTER; filter.addr=(uintptr_t)0;
    if(cuvsCagraSearch(h->res,sp,h->idx,&qt,&nt,&dt,filter)!=CUVS_SUCCESS){rc=15;goto done;}
    if(cudaMemcpy(out_ids,dids,(size_t)nq*topk*4,cudaMemcpyDeviceToHost)!=cudaSuccess){rc=16;goto done;}
    if(out_dist&&cudaMemcpy(out_dist,ddist,(size_t)nq*topk*4,cudaMemcpyDeviceToHost)!=cudaSuccess){rc=17;goto done;}
  }
  if(h->host_graph&&h->host_base&&seeds>0){
    hids=(unsigned*)malloc((size_t)nq*search_k*4); ranked=(Ranked*)malloc(cap*sizeof(Ranked));
    if(!hids||!ranked){rc=18;goto done;}
    if(cudaMemcpy(hids,dids,(size_t)nq*search_k*4,cudaMemcpyDeviceToHost)!=cudaSuccess){rc=16;goto done;}
    if(!h->stamp){h->stamp=(uint32_t*)calloc((size_t)h->n,4); if(!h->stamp){rc=18;goto done;} h->stamp_gen=0;}
    for(long qi=0;qi<nq;qi++){
      uint32_t tag=++h->stamp_gen; if(tag==0){memset(h->stamp,0,(size_t)h->n*4);h->stamp_gen=1;tag=1;}
      const unsigned* row=hids+(size_t)qi*search_k; size_t cnt=0;
      for(long c=0;c<search_k;c++){unsigned id=row[c]; if((long)id<h->n&&h->stamp[id]!=tag){h->stamp[id]=tag;ranked[cnt].id=id;ranked[cnt].d=0;cnt++;}}
      if(cnt>cap) cnt=cap;
      long sd=seeds; if(sd>search_k) sd=search_k;
      for(long c=0;c<sd&&c<(long)cnt;c++){unsigned s=ranked[c].id;const uint32_t* gr=h->host_graph+(size_t)s*h->k;for(int j=0;j<deg;j++){unsigned id=gr[j];if((long)id<h->n&&h->stamp[id]!=tag&&cnt<cap){h->stamp[id]=tag;ranked[cnt].id=id;ranked[cnt].d=0;cnt++;}}}
      const float* qv=query+(size_t)qi*h->dim;
      for(size_t c=0;c<cnt;c++) ranked[c].d=l2(qv,h->host_base+(size_t)ranked[c].id*h->dim,h->dim);
      if(second>0){
        size_t h2=(size_t)second<cnt?(size_t)second:cnt;
        qsort(ranked,cnt,sizeof(Ranked),ranked_less);
        size_t base_cnt=cnt;
        for(size_t c=0;c<h2;c++){unsigned s=ranked[c].id;const uint32_t* gr=h->host_graph+(size_t)s*h->k;for(int j=0;j<deg;j++){unsigned id=gr[j];if((long)id<h->n&&h->stamp[id]!=tag&&cnt<cap){h->stamp[id]=tag;ranked[cnt].id=id;ranked[cnt].d=0;cnt++;}}}
        for(size_t c=base_cnt;c<cnt;c++) ranked[c].d=l2(qv,h->host_base+(size_t)ranked[c].id*h->dim,h->dim);
      }
      qsort(ranked,cnt,sizeof(Ranked),ranked_less);
      for(long j=0;j<topk;j++){if((size_t)j<cnt){out_ids[(size_t)qi*topk+j]=ranked[j].id;if(out_dist)out_dist[(size_t)qi*topk+j]=ranked[j].d;}else{out_ids[(size_t)qi*topk+j]=0;if(out_dist)out_dist[(size_t)qi*topk+j]=INFINITY;}}
    }
  }
done:
  if(sp) cuvsCagraSearchParamsDestroy(sp);
  if(ddist) cudaFree(ddist); if(dids) cudaFree(dids); if(dq) cudaFree(dq);
  free(hids); free(ranked);
  if(rc) fprintf(stderr,"[bf256] search rc=%d: %s\n",rc,cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle){sk_bf_index* h=(sk_bf_index*)handle;if(!h)return;if(h->idx)cuvsCagraIndexDestroy(h->idx);if(h->res)cuvsResourcesDestroy(h->res);free(h->host_base);free(h->host_graph);free(h->stamp);free(h);}

int seekdb_cuvs_serialize(void* handle,const char* filename)
{
  sk_bf_index* h=(sk_bf_index*)handle; if(!h||!filename) return 1;
  if(cuvsCagraSerialize(h->res,filename,h->idx,true)!=CUVS_SUCCESS){fprintf(stderr,"[bf256] serialize: %s\n",cuvsGetLastErrorText());return 1;}
  if(h->host_graph&&h->k>0){
    int64_t hn=h->n,hd=h->dim,hk=h->k;
    uint64_t data_bytes=(uint64_t)(sizeof(uint64_t)+3*sizeof(int64_t))+(uint64_t)hn*hk*4;
    FILE* f=fopen(filename,"ab"); if(!f) return 1;
    uint64_t magic=0x42463235ULL;
    int bad=0;
    bad|=fwrite(&magic,8,1,f)!=1; bad|=fwrite(&hn,8,1,f)!=1; bad|=fwrite(&hd,8,1,f)!=1; bad|=fwrite(&hk,8,1,f)!=1;
    bad|=fwrite(h->host_graph,4,(size_t)hn*hk,f)!=(size_t)hn*hk;
    bad|=fwrite(&data_bytes,8,1,f)!=1; bad|=fwrite(&magic,8,1,f)!=1;
    if(fclose(f)!=0) bad=1;
    if(bad){fprintf(stderr,"[bf256] sidecar write failed\n");return 1;}
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename,long n,long dim)
{
  if(!filename||n<=0||dim<=0) return NULL;
  sk_bf_index* h=(sk_bf_index*)calloc(1,sizeof(*h)); if(!h) return NULL;
  h->n=n; h->dim=dim;
  if(cuvsResourcesCreate(&h->res)!=CUVS_SUCCESS) goto fail;
  if(cuvsCagraIndexCreate(&h->idx)!=CUVS_SUCCESS) goto fail;
  if(cuvsCagraDeserialize(h->res,filename,h->idx)!=CUVS_SUCCESS){fprintf(stderr,"[bf256] deser: %s\n",cuvsGetLastErrorText());goto fail;}
  /* dataset */
  {
    DLManagedTensor dv; memset(&dv,0,sizeof(dv));
    if(cuvsCagraIndexGetDataset(h->idx,&dv)!=CUVS_SUCCESS){fprintf(stderr,"[bf256] get dataset: %s\n",cuvsGetLastErrorText());goto fail;}
    if(!dv.dl_tensor.data||dv.dl_tensor.shape[0]!=n||dv.dl_tensor.shape[1]!=dim) goto fail;
    h->host_base=(float*)malloc((size_t)n*dim*4);
    if(!h->host_base) goto fail;
    if(cudaMemcpy(h->host_base,dv.dl_tensor.data,(size_t)n*dim*4,cudaMemcpyDeviceToHost)!=cudaSuccess) goto fail;
  }
  /* sidecar full graph */
  {
    FILE* f=fopen(filename,"rb"); if(!f) goto fail;
    if(fseek(f,0,SEEK_END)!=0){fclose(f);goto fail;}
    long long fs=ftell(f); if(fs<=32){fclose(f);goto fail;}
    uint64_t sz=0,tmagic=0; int ok=0;
    if(fseek(f,(long)(fs-16),SEEK_SET)==0&&fread(&sz,8,1,f)==1&&fread(&tmagic,8,1,f)==1&&tmagic==0x42463235ULL&&sz>0&&(long long)sz+16<=fs){
      if(fseek(f,(long)(fs-16-(long long)sz),SEEK_SET)==0){
        uint64_t magic=0; int64_t hn=0,hd=0,hk=0;
        if(fread(&magic,8,1,f)==1&&fread(&hn,8,1,f)==1&&fread(&hd,8,1,f)==1&&fread(&hk,8,1,f)==1&&magic==0x42463235ULL&&hn==n&&hd==dim&&hk>0&&hk<=4096){
          h->host_graph=(uint32_t*)malloc((size_t)hn*hk*4);
          if(h->host_graph&&fread(h->host_graph,4,(size_t)hn*hk,f)==(size_t)hn*hk){h->k=hk;ok=1;}
          else{free(h->host_graph);h->host_graph=NULL;}
        }
      }
    }
    fclose(f);
    if(!ok) fprintf(stderr,"[bf256] sidecar missing; expansion disabled\n");
  }
  if(!h->k){int64_t gd=0;if(cuvsCagraIndexGetGraphDegree(h->idx,&gd)==CUVS_SUCCESS)h->k=gd;h->index_k=h->k;}
  return h;
fail:
  if(h->idx) cuvsCagraIndexDestroy(h->idx);
  if(h->res) cuvsResourcesDestroy(h->res);
  free(h->host_base); free(h->host_graph); free(h);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base,long n,long dim,const float* query,long nq,long topk,unsigned* out_ids)
{
  void* h=seekdb_cuvs_build(base,n,dim); if(!h) return 10;
  int rc=seekdb_cuvs_search(h,query,nq,topk,out_ids,NULL);
  seekdb_cuvs_free(h); return rc;
}
