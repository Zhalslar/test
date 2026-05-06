def a(b):
 try:
  if b>0:return a(b-1)
  else:return "味道如何？"
 except:pass
x=lambda y:print(a(y))
if __name__=="__main__":x(500)
# 毫无意义的改动
print('我是垃圾')
