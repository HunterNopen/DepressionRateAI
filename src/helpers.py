from re import sub
from random import randrange

def to_snake_case(s):
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
        sub('([A-Z]+)', r' \1',
        s.replace('-', ' '))).split()).strip().lower()

def sleep_col_mediate(s):
    match s:
        case "Less than 5 hours":
            return "2-4 hrs"
        case "5-6 hours":
            return "4-6 hrs"
        case "7-8 hours" | "More than 8 hours" | _:
            return "7-8 hrs"
        
def sleep_col_mediate_to_int(s):
    match s:
        case "2-4 hrs":
            return 0
        case "4-6 hrs":
            return 1
        case "7-8 hrs" | _:
            return 2
        
        
def study_hours_col_mediate(s):
    match s:
        case 1:
            return f'{randrange(0, 3)} hrs'
        case 2:
            return f'{randrange(3, 6)} hrs'
        case 3:
            return f'{randrange(6, 9)} hrs'
        case 4:
            return f'{randrange(9, 11)} hrs'
        case 5:
            return f'{randrange(11, 13)} hrs'
        case _:
            return 6
        
def study_hours_col_mediate_to_int(s):
    match s:
        case _ if s in range(0, 3):
            return 1
        case _ if s in range(3, 6):
            return 2
        case _ if s in range(6, 9):
            return 3
        case _ if s in range(9, 11):
            return 4
        case _ if s in range(11, 13):
            return 5
        case _:
            return 2
        
def depression_col_mediate(s):
    match s:
        case _ if s in range(0, 3):
            return 0
        case _ if s in range(3, 6):
            return 1
        case _:
            return 1