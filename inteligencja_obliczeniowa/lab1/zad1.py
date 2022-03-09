import math


def isPrime(num: int):
    for divisor in range(2, math.floor(math.sqrt(num)) + 1):
        if num % divisor == 0:
            return False
    return True


def arePrimes(nums: []):
    outcome = []
    for num in nums:
        if isPrime(num):
            outcome.append(num)
    return outcome
