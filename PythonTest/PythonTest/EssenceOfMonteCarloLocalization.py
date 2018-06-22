p = [0.2, 0.2, 0.2, 0.2, 0.2]
#p = [0, 1, 0, 0, 0]
#world = ['green', 'red', 'red', 'green', 'green'] #Replace Z with measurements vector and assume the robot is going to sense red, then green
world = ['green', 'red', 'green', 'green', 'green'] #Replace Z with measurements vector and assume the robot is going to sense red, then green
#measurements = ['red'] #['red', 'green']
measurements = ['red', 'green']

pHit = 0.6
pMiss = 0.2

#Add exact probability
pExact = 0.8
#Add overshoot probability
pOvershoot = 0.1
#Add undershoot probability
pUndershoot = 0.1

motions = [1, 1]


def sense(p, Z):

    q = [ ]

    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))

    s = sum(q)

    for i in range (len(p)):
        q[i] = q[i]/s

    return q #Division by the sum gives us our normalized distribution


def move(p, U):

    q= [ ] #Start with empty list

    for i in range(len(p)): #Go through all the elements in p
        #Construct q element by element by accessing the corresponding p which is shifted by U
        s = pExact * p[(i-U) % len(p)]
        s = s + pOvershoot * p[(i-U-1) % len(p)]
        s = s + pUndershoot * p[(i-U+1) % len(p)]
        q.append(s)

    return q


#As often as there are measurements use the following for loop
#for k in range(len(measurements)): #Grab the k element, apply to the current belief, p, and update that belief into itself
#    p = sense(p, measurements[k])

#print move(p, 2) #run this twice and get back the uniform distribution:

#p = move(p, 1)
#p = move(p, 1)

#for k in range(1000):
#    p = move(p, 1)
#    print p

for k in range(len(measurements)):
    p = sense(p, measurements[k])
    p = move(p, motions[k])

print p