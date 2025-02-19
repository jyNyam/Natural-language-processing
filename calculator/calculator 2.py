#  calculator 개선하여 한번에 처리하는 계산기

import ast

def safe_eval(expression):
    try:
        tree = ast.parse(expression, mode = 'eval')

        allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, 
                         ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError("지원하지 않는 연산자입니다.")
        
        return eval(expression)
    except:
        return "잘못된 입력입니다."

def calculator():
    while True:
        expression = input("계산할 식을 입력하시오(종료: 'q'): ")

        if expression.lower() == "q":
            print("계산기를 종료합니다.")
            break

        result = safe_eval(expression)
        print(f"결과: {result}\n")

calculator()