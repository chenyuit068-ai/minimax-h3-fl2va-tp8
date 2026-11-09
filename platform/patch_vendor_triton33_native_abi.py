"""Pair the official HCU backend's native HIP structs with the matching v2 ABI."""
import difflib,sys
from pathlib import Path
root=Path(sys.argv[1]);driver=root/'triton/backends/hcu/driver.py';original=driver.read_text()
assert 'H3_TRITON_NATIVE_PROPERTIES_V2' not in original
needle='include_dir = [os.path.join(dirname, "../hcu","include")]'
replacement='''_native_hip_include = os.environ.get("TRITON_HIP_INCLUDE_PATH")
include_dir = ([_native_hip_include] if _native_hip_include else []) + [os.path.join(dirname, "../hcu", "include")]'''
function='def compile_module_from_src(src, name):\n'
replacement_function='''def compile_module_from_src(src, name):
    if _native_hip_include:
        headers = [Path(_native_hip_include) / "hip" / name for name in ("hip_runtime.h", "hip_runtime_api.h")]
        fingerprint = hashlib.sha256(b"".join(header.read_bytes() for header in headers)).hexdigest()
        src = "// Native HIP ABI headers: " + fingerprint + "\\n" + src
    if name == "hip_utils" and os.environ.get("TRITON_HIP_PROPERTIES_V2") == "1":
        if not _native_hip_include:
            raise RuntimeError("Native v2 properties require matching DTK headers")
        src = "#define H3_TRITON_NATIVE_PROPERTIES_V2 1\\n" + src
'''
assert original.count(needle)==1 and original.count(function)==1
patched=original.replace(needle,replacement,1).replace(function,replacement_function,1)
compile(patched,str(driver),'exec')
driver.with_suffix('.py.h3-vendor-original').write_text(original);driver.write_text(patched)
(root/'host-header-search.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='a/triton/backends/hcu/driver.py',tofile='b/triton/backends/hcu/driver.py')))
c=root/'triton/backends/hcu/driver.c';original_c=c.read_text()
assert original_c.count('hipDeviceProp_tR0000')==2
patched_c=original_c.replace('hipDeviceProp_tR0000','H3HipDeviceProperties')
needle_c='#include <hip/hip_deprecated.h>'
assert patched_c.count(needle_c)==1
patched_c=patched_c.replace(needle_c,'''#include <stdbool.h>
#ifdef H3_TRITON_NATIVE_PROPERTIES_V2
typedef hipDeviceProp_t H3HipDeviceProperties;
#else
#include <hip/hip_deprecated.h>
typedef hipDeviceProp_tR0000 H3HipDeviceProperties;
#endif''',1)
symbol_needle='  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)\n\n  return true;'
symbol_replacement='''  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)
#ifdef H3_TRITON_NATIVE_PROPERTIES_V2
  dlerror();
  void *propertiesFn = dlsym(lib, "hipGetDeviceProperties_v2");
  if (!propertiesFn) {
    PyErr_SetString(PyExc_RuntimeError, "DTK hipGetDeviceProperties_v2 is unavailable");
    dlclose(lib);
    return false;
  }
  *(void **)&hipSymbolTable.hipGetDeviceProperties = propertiesFn;
#endif
  return true;'''
assert patched_c.count(symbol_needle)==1
patched_c=patched_c.replace(symbol_needle,symbol_replacement)
c.with_suffix('.c.h3-vendor-original').write_text(original_c);c.write_text(patched_c)
(root/'host-properties-abi.patch').write_text(''.join(difflib.unified_diff(original_c.splitlines(True),patched_c.splitlines(True),fromfile='a/triton/backends/hcu/driver.c',tofile='b/triton/backends/hcu/driver.c')))
print('HCU_HOST_NATIVE_HEADER_AND_V2_PROPERTY_ABI_PATCHED',flush=True)
