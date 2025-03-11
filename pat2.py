n = int(input("Enter a number: "))
r = (n*2)-1 #n+n-1
for i in range(1,r+1):
    for j in range(1,r+1):
        for p in range(1,n+1):
            if(i==p or j==p or i==n+(n-p) or j==n+(n-p)):
                print(f"{n-(p-1)} ",end = " ")
                break
    print()