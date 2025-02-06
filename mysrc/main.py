# coding: utf-8
from Research import *

def main():

    baranov_result = threaded_research(r_values=[28],gap_number=20)
    my_result = threaded_research(r_values=[28,25,27], gap_number=20)
    with open("output.txt", "w") as file:
        file.write(str(baranov_result)+'\n')
        file.write(str(my_result))
if __name__ == '__main__':
    main()