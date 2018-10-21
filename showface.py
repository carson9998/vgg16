import cPickle
import pylab
read_file=open('/home/carson/one/olivettifaces.pkl','rb')  
faces=cPickle.load(read_file)
read_file.close() 
img1=faces[398].reshape(57,47)
pylab.imshow(img1)
pylab.gray()
pylab.show()