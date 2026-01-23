
```shell
>>> Testing N = 262144 (2 MB data)
Verification PASSED: GPU result = 194.38, CPU result = 194.381
Verification PASSED: GPU result = 194.381, CPU result = 194.381
Verification PASSED: GPU result = 194.381, CPU result = 194.381

>>> Testing N = 524288 (4 MB data)
Verification PASSED: GPU result = 123.325, CPU result = 123.325
Verification PASSED: GPU result = 123.325, CPU result = 123.325
Verification PASSED: GPU result = 123.325, CPU result = 123.325

>>> Testing N = 1048576 (8 MB data)
Verification PASSED: GPU result = 207.921, CPU result = 207.92
Verification PASSED: GPU result = 207.92, CPU result = 207.92
Verification PASSED: GPU result = 207.92, CPU result = 207.92

>>> Testing N = 134217728 (1024 MB data)
Verification PASSED: GPU result = 7146.49, CPU result = 7146.46
Verification PASSED: GPU result = 7146.46, CPU result = 7146.46
Verification PASSED: GPU result = 7146.46, CPU result = 7146.46

>>> Testing N = 268435456 (2048 MB data)
Verification PASSED: GPU result = -7979.41, CPU result = -7979.46
Verification PASSED: GPU result = -7979.46, CPU result = -7979.46
Verification PASSED: GPU result = -7979.46, CPU result = -7979.46

>>> Testing N = 536870912 (4096 MB data)
Verification FAILED: GPU result = 914.927, CPU result = 914.948
Verification PASSED: GPU result = 914.948, CPU result = 914.948
Verification PASSED: GPU result = 914.949, CPU result = 914.948

>>> Testing N = 805306368 (6144 MB data)
Verification FAILED: GPU result = -222.414, CPU result = -222.401
Verification FAILED: GPU result = -222.397, CPU result = -222.401
Verification FAILED: GPU result = -222.396, CPU result = -222.401

>>> Testing N = 1073741824 (8192 MB data)
Verification FAILED: GPU result = -273.733, CPU result = -273.756
Verification PASSED: GPU result = -273.758, CPU result = -273.756
Verification FAILED: GPU result = -273.753, CPU result = -273.756

```