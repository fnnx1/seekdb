#include "seekdb_cuvs_bridge.h"
#include <cuvs/cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
static long long used(){ size_t f=0,t=0; hipMemGetInfo(&f,&t); return (long long)(t-f); }
int main(){
  // platform "small-n audit" followed by the real build, in one process
  for (int phase=0; phase<2; ++phase){
    long n = phase==0 ? 1000 : 20000; long dim = phase==0 ? 8 : 64;
    std::vector<float> base(n*dim);
    for(long r=0;r<n;++r) for(long c=0;c<dim;++c) base[r*dim+c]=sinf(float(r*7+c*13));
    long long u0=used();
    void* h = seekdb_cuvs_build(base.data(), n, dim);
    long long u1=used();
    unsigned ids[10]; float d[10];
    int rc = h ? seekdb_cuvs_search(h, base.data(), 1, 10, ids, d) : -1;
    long long u2=used();
    if(h) seekdb_cuvs_free(h);
    printf("phase%d n=%ld dim=%ld rc=%d used: before=%.3f after_build=%.3f after_search=%.3f after_free=%.3f GB\n",
           phase, n, dim, rc, u0/1e9, u1/1e9, u2/1e9, used()/1e9);
    if(!h || rc!=0 || ids[0]!=0) return 1;
  }
  return 0;
}
