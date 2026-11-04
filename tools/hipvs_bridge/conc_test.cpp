#include "seekdb_cuvs_bridge.h"
#include <cuvs/cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>
int main(){
  const long n=10000, dim=32, nq=8, topk=10;
  std::vector<float> base(n*dim), q(nq*dim);
  for(long r=0;r<n;++r) for(long c=0;c<dim;++c) base[r*dim+c]=sinf(float(r*7+c*13));
  for(long r=0;r<nq;++r) for(long c=0;c<dim;++c) q[r*dim+c]=sinf(float(r*31+c*3));
  void* h = seekdb_cuvs_build(base.data(), n, dim);
  if(!h){ printf("build failed\n"); return 2; }
  unsigned ref[nq*topk];
  if(seekdb_cuvs_search(h, q.data(), nq, topk, ref, nullptr)) { printf("ref search failed\n"); return 3; }
  std::atomic<int> bad{0};
  std::vector<std::thread> th;
  for(int t=0;t<4;++t) th.emplace_back([&](){
    for(int it=0; it<20; ++it){
      unsigned out[nq*topk];
      if(seekdb_cuvs_search(h, q.data(), nq, topk, out, nullptr) != 0) { bad++; continue; }
      if(memcmp(out, ref, sizeof(ref)) != 0) bad++;
    }
  });
  for(auto& x: th) x.join();
  printf("concurrency: 4 threads x 20 searches, mismatches=%d\n", bad.load());
  seekdb_cuvs_free(h);
  return bad.load() ? 1 : 0;
}
