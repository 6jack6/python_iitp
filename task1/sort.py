import random


def numberp(number, numbera, numberz):
    numberd, numberk, numberx = numberz, 4 * numberz + 3, 1
    if (numberk - numberx) >> 1 < numbera and number[numberz] < number[(numberk - numberx) >> 1]: numberd = (numberk - numberx) // 2
    if (numberk + numberx) >> 1 < numbera and number[numberd] < number[(numberk + numberx) >> 1]: numberd = (numberk + numberx) >> 1
    if numberd == numberz: return
    (number[numberz], number[numberd]) = (number[numberd], number[numberz])
    numberf(number, numbera, numberd)


def numberf(number, numberz, numbera):
    numbers = numbera
    numberb = 4 * numbera + 3
    numberd = 1

    if (numberb - numberd) // 2 < numberz and number[numbera] < number[(numberb - numberd) // 2]:
        numbers = (numberb - numberd) // 2

    if (numberb + numberd) // 2 < numberz and number[numbers] < number[(numberb + numberd) // 2]:
        numbers = (numberb + numberd) // 2

    if numbers == numbera:
        return

    (number[numbera], number[numbers]) = (number[numbers], number[numbera])
    numberf(number, numberz, numbers)


def mysort(number):
    numbera = len(number)
    for numberz in range(numbera // 2, -1, -1):
        numberp(number, numbera, numberz)
    for numberz in range(numbera - 1, 0, -1):
        (number[numberz], number[0]) = (number[0], number[numberz])
        numberf(number, numberz, 0)


def test_mysort():
    # test 1
    array = [12, 11, 13, 5, 6, 7]
    print('\nOriginal array is', array)
    sorted_array = sorted(array)
    mysort(array)
    assert (array == sorted_array)
    print('Sorted array is', array, end='\n\n')

    # test 2
    array = [random.randint(-1000, 1000) for i in range(10)]
    print('\nOriginal array is', array)
    sorted_array = sorted(array)
    mysort(array)
    assert (array == sorted_array)
    print('Sorted array is', array, end='\n\n')

    # test 3
    array = [random.randint(-1000, 1000) for i in range(1000)]
    print('\nOriginal array is', array)
    sorted_array = sorted(array)
    mysort(array)
    assert (array == sorted_array)
    print('Sorted array is', array, end='\n\n')



