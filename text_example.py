import time
import datetime


print(time.localtime(1461161056.2800557613))
print(time.localtime(1461161056.3000824451))
print(time.localtime(1461161056.3200588226))
print(time.localtime(1461161056.3400502205))


# current = datetime.datetime.now()
current = time.time()
# milly_after = current + datetime.timedelta(milliseconds=16.6)
milly_after = current +0.016
print(type(current))
print(milly_after)

f = open("연습용.txt", 'w')
stamp = time.time()

for i in range(1, 11):
    print(stamp, type(stamp))
    data = "{:0>5}번째 줄입니다. {}\n".format(i,stamp)
    f.write(data)
    stamp += 0.016
    # stamp =



f.close()