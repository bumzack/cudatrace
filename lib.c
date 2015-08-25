#define _GNU_SOURCE

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>

enum cudaError        {cudaError_dummy};
enum cudaMemcpyKind   {cudaMemcpyKind_dummy};
enum cublasStatus     {cublasStatus_dummy};
enum cublasOperation  {cublasOperation_dummy};

typedef enum cudaError cudaError_t;
typedef enum cublasStatus cublasStatus_t;
typedef enum cublasOperation cublasOperation_t;

struct CUstream_st;
struct cublasContext;

typedef struct CUstream_st *cudaStream_t;
typedef struct cublasContext *cublasHandle_t;

#define CUDATRACE_DECLARE(ret_ty, sym, ...) \
    ret_ty (*real_##sym)(__VA_ARGS__) = NULL; \
    ret_ty sym(__VA_ARGS__)

#define CUDATRACE_INTERPOSE(sym) \
    ({ \
      if (real_##sym == NULL) { \
        real_##sym = dlsym(RTLD_NEXT, #sym); \
      } \
      assert(NULL != real_##sym && "FATAL: failed to interpose " #sym); \
    })

cudaError_t (*real_cudaMemcpy)(
    void *,
    void *,
    size_t,
    enum cudaMemcpyKind) = NULL;
cudaError_t cudaMemcpy(
    void *dst,
    void *src,
    size_t size,
    enum cudaMemcpyKind kind)
{
  CUDATRACE_INTERPOSE(cudaMemcpy);
  fprintf(stderr, "TRACE: cudaMemcpy %p %p %lu %d\n",
      dst, src, size, kind);
  return real_cudaMemcpy(
      dst, src, size, kind);
}

cudaError_t (*real_cudaMemcpyAsync)(
    void *,
    void *,
    size_t,
    enum cudaMemcpyKind,
    cudaStream_t) = NULL;
cudaError_t cudaMemcpyAsync(
    void *dst,
    void *src,
    size_t size,
    enum cudaMemcpyKind kind,
    cudaStream_t stream)
{
  CUDATRACE_INTERPOSE(cudaMemcpyAsync);
  fprintf(stderr, "TRACE: cudaMemcpyAsync %p %p %lu %d %p\n",
      dst, src, size, kind, stream);
  return real_cudaMemcpyAsync(
      dst, src, size, kind, stream);
}

CUDATRACE_DECLARE(cublasStatus_t, cublasSgemm_v2,
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc)
{
  CUDATRACE_INTERPOSE(cublasSgemm_v2);
  fprintf(stderr, "TRACE: cublasSgemm_v2 %p %d %d %d %d %d %p %p %d %p %d %p %p %d\n",
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return real_cublasSgemm_v2(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

CUDATRACE_DECLARE(cublasStatus_t, cublasSgemv_v2,
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m,
    int n,
    const float *alpha,
    const float *A,
    int lda,
    const float *x,
    int incx,
    const float *beta,
    float *y,
    int incy)
{
  CUDATRACE_INTERPOSE(cublasSgemv_v2);
  fprintf(stderr, "TRACE: cublasSgemv_v2 %p %d %d %d %p %p %d %p %d %p %p %d\n",
      handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  return real_cublasSgemv_v2(
      handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

CUDATRACE_DECLARE(cublasStatus_t, cublasSaxpy_v2,
    cublasHandle_t handle,
    int n,
    const float *alpha,
    const float *x,
    int incx,
    float *y,
    int incy)
{
  CUDATRACE_INTERPOSE(cublasSaxpy_v2);
  fprintf(stderr, "TRACE: cublasSaxpy_v2 %p %d %p %p %d %p %d\n",
      handle, n, alpha, x, incx, y, incy);
  return real_cublasSaxpy_v2(
      handle, n, alpha, x, incx, y, incy);
}

CUDATRACE_DECLARE(cublasStatus_t, cublasSscal_v2,
    cublasHandle_t handle,
    int n,
    const float *alpha,
    float *x,
    int incx)
{
  CUDATRACE_INTERPOSE(cublasSscal_v2);
  fprintf(stderr, "TRACE: cublasSscal_v2 %p %d %p %p %d\n",
      handle, n, alpha, x, incx);
  return real_cublasSscal_v2(
      handle, n, alpha, x, incx);
}

CUDATRACE_DECLARE(cublasStatus_t, cublasSdot_v2,
    cublasHandle_t handle,
    int n,
    const float *x,
    int incx,
    const float *y,
    int incy,
    float *result)
{
  CUDATRACE_INTERPOSE(cublasSdot_v2);
  fprintf(stderr, "TRACE: cublasSdot_v2 %p %d %p %d %p %d %p\n",
      handle, n, x, incx, y, incy, result);
  return real_cublasSdot_v2(
      handle, n, x, incx, y, incy, result);
}

CUDATRACE_DECLARE(cublasStatus_t, cublasSasum_v2,
    cublasHandle_t handle,
    int n,
    const float *x,
    int incx,
    float *result)
{
  CUDATRACE_INTERPOSE(cublasSasum_v2);
  fprintf(stderr, "TRACE: cublasSasum_v2 %p %d %p %d %p\n",
      handle, n, x, incx, result);
  return real_cublasSasum_v2(
      handle, n, x, incx, result);
}

CUDATRACE_DECLARE(cublasStatus_t, cublasScopy_v2,
    cublasHandle_t handle,
    int n,
    const float *x,
    int incx,
    float *y,
    int incy)
{
  CUDATRACE_INTERPOSE(cublasScopy_v2);
  fprintf(stderr, "TRACE: cublasScopy_v2 %p %d %p %d %p %d\n",
      handle, n, x, incx, y, incy);
  return real_cublasScopy_v2(
      handle, n, x, incx, y, incy);
}
