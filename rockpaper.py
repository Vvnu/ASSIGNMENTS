import random
choices = ['R', 'P', 'S'] 
choice_map = {'R': 'Rock', 'P': 'Paper', 'S': 'Scissors'}

lst = []  

user_choice = input("Enter your choice (R for Rock, P for Paper, S for Scissors): ").upper()

if user_choice not in choices:
    print("Invalid choice. Please choose R, P, or S.")
else:
    lst.append(user_choice)  

    computer_choice = random.choice(choices)
    lst.append(computer_choice)

    print(f"You chose {choice_map[lst[0]]}")
    print(f"Computer chose {choice_map[lst[1]]}")

    if lst[0] == lst[1]:
        print("It's a tie!")
    elif (lst[0] == 'R' and lst[1] == 'S') or \
         (lst[0] == 'P' and lst[1] == 'R') or \
         (lst[0] == 'S' and lst[1] == 'P'):
        print("You win!")
    else:
        print("Computer wins!")
