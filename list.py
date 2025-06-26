list1 = ["apple","Banana","Grapes"]
for x in list1:
    print(x)




x = int(input("Enter the marks : "))
if((x>=33) and  (x<=100)):
    print("Student is Pass")
elif(x>=101):
    print("Student data is invalid")
elif(x<=32 and x>=1):
    print("student is failed")