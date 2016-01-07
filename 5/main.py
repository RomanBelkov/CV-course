import numpy
from loader import MNIST

# consts
digitA, digitB = 4, 2  # binary classification
color_threshold = 170
img_size = 28 * 28
iterations = 5
nu = 0.01


def label_class(l):
    return -1 if l == digitA else (1 if l == digitB else 0)


def color_binarization(img):
    return numpy.array([(0 if pix < color_threshold else 1) for pix in img])


def do_train(w, imgs, classes):
    n = len(imgs)
    res = sum([(classes[i] * imgs[i]) / (1.0 + numpy.exp(numpy.dot(classes[i], w.T) * imgs[i])) for i in xrange(n)])
    return (-1.0 / n) * res


mndata = MNIST('data')
mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'

print "Params: "
print "Digits: " + str(digitA) + " -- -1 " + str(digitB) + " -- 1"
print "Iterations: ", iterations
print "Step (nu): ", nu

print "Loading data..."
mndata.load_training()
mndata.load_testing()

print "Training data count:", len(mndata.train_images)
print "Testing data count:", len(mndata.test_images)

[(train_imgs, train_classes), (test_imgs, test_classes)] = [
    zip(*[(i, l) for (i, l) in zip(imgs, map(label_class, lbls)) if l == -1 or l == 1])
    for (imgs, lbls) in [(mndata.train_images, mndata.train_labels), (mndata.test_images, mndata.test_labels)]]

train_imgs = map(color_binarization, train_imgs)
test_imgs = map(color_binarization, test_imgs)

w = numpy.array([0.0 for _ in xrange(img_size)])

print "Training..."
for _ in xrange(iterations):
    w -= nu * do_train(w.copy(), train_imgs, train_classes)

TP, TN, FP, FN = 0, 0, 0, 0

print "Testing..."
results = [1.0 / (1.0 + numpy.exp(-numpy.dot(i, w.T))) for i in test_imgs]
results = [(1 if x > 0.5 else -1) for x in results]

successes = [res for (res, expected) in zip(results, test_classes) if res == expected]
fails = [res for (res, expected) in zip(results, test_classes) if res != expected]

TP = sum([1 for x in successes if x == 1])
TN = sum([1 for x in successes if x == -1])
FP = sum([1 for x in fails if x == 1])
FN = sum([1 for x in fails if x == -1])

print "TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN

print "Precision:", float(TP) / (TP + FP)
print "Recall:", float(TP) / (TP + FN)
