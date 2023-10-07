from multiprocessing import Pool

def f(n):
  return n*n

if __name__ == '__main__':
  array = [1,2,3,4,5]
  p = Pool()
  result = p.map(f,array)
  p.close()
  p.join()
  print(result)
