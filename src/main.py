#
#   main.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from managers.parser_manager import ParserManager

if __name__ == '__main__':
    parser: ParserManager = ParserManager()
    parser.parse()
    print(f"Flag: '{parser.flag}'\nPath: '{parser.path}'")
