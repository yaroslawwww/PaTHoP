# coding: utf-8
from Research import *

def main():

    baranov_result = threaded_research(r_values=[28],gap_number=20)
    my_result = threaded_research(r_values=[28,25,27,29,30,31], gap_number=20)
    with open("output.txt", "w") as file:
        file.write("Результат для одного ряда")
        file.write(str(round(baranov_result,5))+'\n')
        file.write("Результат для 6 рядов")
        file.write(str(round(my_result,5)))
if __name__ == '__main__':
    main()