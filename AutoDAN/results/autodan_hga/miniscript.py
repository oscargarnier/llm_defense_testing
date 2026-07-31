string = ""
for i in range(512):
    string += f"{i}: \n"

with open("output.txt", "w") as f:
    f.write(string)