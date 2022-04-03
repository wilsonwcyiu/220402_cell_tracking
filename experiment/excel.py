

from openpyxl import Workbook       # pip install openpyxl


if __name__ == '__main__':

    workbook = Workbook()
    sheet = workbook.active

    sheet["A1"] = "hello"
    sheet["B1"] = "world!"

    workbook.save(filename="D:/tmp/hello_world.xlsx")
