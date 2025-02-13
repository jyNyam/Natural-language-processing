def matrix(model):
    # 세모
    if model == 1:
        # for i in range(1, 11, 2):
        #    print('{:^10}'.format('*' * i))
        for i in range(1, 6):
            for j in range(5 - i):
                print(' ', end='')
            for j in range(1, i * 2, 1):
                print('*', end='')
            print('')
    # 네모
    elif model == 2:
        for i in range(6):
            print('{:^10}'.format('*' * 9))
        print('')
        for i in range(1, 7):
            if i == 1 or i == 6:
                for j in range(9):
                    print('*', end='')
                print('')
            else:
                for j in range(1, 10):
                    if j == 1 or j == 9:
                        print('*', end='')
                    else:
                        print(' ', end='')

                print('')
    # 다이아
    elif model == 3:
        # for i in range(1, 11, 2):
        #    print('{:^10}'.format('*' * i))
        # for i in range(9, 0, -2):
        #    print('{:^10}'.format('*' * i))
        for i in range(1, 6):
            for j in range(5 - i):
                print(' ', end='')
            for j in range(1, i * 2, 1):
                print('*', end='')
            print('')
        for i in range(5):
            for j in range(i):
                print(' ', end='')
            for j in range(10, 1 + i * 2, -1):
                print('*', end='')
            print('')
