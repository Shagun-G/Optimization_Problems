file1 = open("fullproblist", 'r')

Lines = file1.readlines()

file1.close()

problem = []
invalue = []
for line in Lines:
    fields  = line.split()
    problem.append(fields[0])
    invalue.append(fields[1])
    
problems_parameters = {}
with open("cutest_parameters.txt", 'w') as f:
    for i in range(len(problem)):
        theproblem = problem[i]
        thevalue   = invalue[i]
        if ( theproblem[0] == "#" ):
            continue
        line = f"{theproblem} = '{thevalue}'"
        f.write(line + '\n')
# for i in range(len(problem)):
#     theproblem = problem[i]
#     thevalue   = invalue[i]
#     if ( theproblem[0] == "#" ):
#         continue
#     line = f"{theproblem} = '{thevalue}'"

