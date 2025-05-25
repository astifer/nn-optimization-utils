from .test_autograd import test_autograd_exp_plus_cos
from .test_rmsnorm import test_check_rmsnorm
from .test_lion import test_lion_optimizer

if __name__ == '__main__':
    print(f"Testing RMSNorm...")
    test_check_rmsnorm()
    print()
    print(f"Testing AutoGrad...")
    test_autograd_exp_plus_cos()
    print()
    print(f"Testing Lion Optimizer...")
    test_lion_optimizer()