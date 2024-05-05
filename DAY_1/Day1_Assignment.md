```md
# Question 1:

```python
x = """
    Hello! This is the output of solutions.py
    and this message is 
    present in three different lines!
"""

print(x)
```

This code block defines a multi-line string `x`, containing a message spanning three lines. The `print(x)` statement outputs the contents of the string `x`.

---

# Question 2:

```python
def get_and_display_details():
    studentid = input("Enter student id: ")
    name = input("Enter name: ")
    branch = input("Enter branch: ")
    university = input("Enter university: ")
    print("Here are the details: ")
    print(studentid, name, branch, university)

get_and_display_details()
```

This code defines a function `get_and_display_details()` that prompts the user to input student details (student ID, name, branch, and university) and then displays these details.

---

# Question 3:

```python
import solution3 as s

# a.)
a = s.area_of_rectangle(5,2)
b = s.area_of_square(5)

print("Area of rectangle: ", a, "Area of square:", b)


# b.) 
d = s.area_of_square(25)
print("Area of square: ", d)

# c.) 
e = s.area_of_rectangle(325, 20)
print("Area of rectangle: ", e)
```

This code imports a module named `solution3` as `s`. It then calls functions from this module to calculate the area of rectangles and squares.

- Part a) calculates the area of a rectangle with dimensions 5 and 2, and the area of a square with side length 5.
- Part b) calculates the area of a square with side length 25.
- Part c) calculates the area of a rectangle with dimensions 325 and 20.
