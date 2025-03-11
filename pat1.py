rows = int(input("Enter the number of rows: "))  # Number of rows in the pattern
start_char = chr(64 + sum(range(rows + 1)))  # Starting character of the pattern for nth row

# Convert the starting character to its ASCII value
ascii_val = ord(start_char)

for i in range(rows):
    # Print spaces for indentation
    print(' ' * (2 * i), end='')
    
    # Print the characters in decreasing order for the current row
    for j in range(rows - i):
        print(chr(ascii_val), end=' ')
        ascii_val -= 1
    
    # Move to the next line
    print()