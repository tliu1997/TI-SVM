import numpy as np
import time
from Model import SVM, TISVM, RISVM, LOCSVM, LTISVM, LRISVM, TIRISVM, LTIRISVM, RIISVM
from DataLoader import load_data, train_valid_split
from kNN import KNN
import ctypes

# train_list = [100, 200, 500]
# val_list = [20, 40, 100]
# test_list = [20800, 20800, 20800]
train_list = [100]
val_list = [20]
test_list = [20800]


def result_record(file, train_acc, valid_acc, test_acc, svclassifier):
    '''
	Args:
		file: file that will record result
		train_acc: training accuracy
		valid_acc: validation accuracy
		test_acc: test accuracy
		svclassifier: trained svm classifier
	Returns:
		No return. Write accuracy values in the file.
	'''
    file.write('Training: \n')
    file.write('accuracy: {:.3f}% \n'.format(train_acc * 100))
    file.write('Validation: \n')
    file.write('accuracy: {:.3f}% \n'.format(valid_acc * 100))
    file.write('Test: \n')
    file.write('accuracy: {:.3f}% \n'.format(test_acc * 100))
    # file.write('The total number of support vectors: \n')
    # file.write(str(np.sum(svclassifier.n_support_)) + '\n')
    file.write('\n')


def main():
    # ----------translational-invariant kernel----------
    # data preprocessing (EMNIST)
    x_train, y_train, x_test, y_test = load_data('EMNIST')
    x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

    # SVM with polynomial kernel (degree=8)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = SVM(degree=8)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write('SVM with polynomial kernel (degree=8, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # SVM with translational-invariant kernel (degree = 8, invariant distance)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = TISVM(degree=8)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write('SVM with translational-invariant kernel (degree=8, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # SVM with locality kernel (degree=8)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = LOCSVM(degree=8, filter=9)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write('SVM with locality kernel (degree=8, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # SVM with locality and rotational-invariant kernel (degree = 8)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = LRISVM(degree=8, filter=5)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write('SVM with locality and rotational-invariant kernel (degree=8, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # SVM with rotational-invariant kernel (degree = 8)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = RISVM(degree=8)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write('SVM with rotational-invariant kernel (degree=8, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # SVM with rotational-invariant kernel (degree = 8, II)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = RIISVM(degree=8)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write('SVM with rotational-invariant kernel (degree=8, II, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # SVM with translational & rotational-invariant kernel (degree = 8)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = TIRISVM(degree=8)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write('SVM with translation & rotational-invariant kernel (degree=8, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # SVM with locality and translational-invariant kernel (degree = 8)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = LTISVM(degree=8, filter=7)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write('SVM with locality and translational-invariant kernel (degree=8, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # SVM with translational & rotational-invariant & locality kernel (degree = 8)
    for idx in range(len(train_list)):
        time1 = time.time()
        model = LTIRISVM(degree=8, filter=7)
        svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
        time2 = time.time()
        with open('result record.txt', 'a') as f:
            f.write(
                'SVM with translation & rotational-invariant & locality kernel (degree=8, EMNIST Letters dataset) \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, svclassifier)

    # def oneSideTD(img1, img2):
    #     d1 = ctypes.CDLL("./td.so").tangentDistance
    #     d1.restype = ctypes.c_double
    #     c_img1 = (ctypes.c_double * len(img1))(*img1)
    #     c_img2 = (ctypes.c_double * len(img2))(*img2)
    #     choice = [1, 1, 0, 0, 0, 1, 0, 0, 0]
    #     choice = np.array(choice)
    #     choice = (ctypes.c_int * len(choice))(*choice)
    #     background = (ctypes.c_double)(0.0)
    #     return d1(c_img1, c_img2, ctypes.c_int(28), ctypes.c_int(28), choice, background)

    def twoSideTD(img1, img2):
        d2 = ctypes.CDLL("./td.so").twoSidedTangentDistance
        d2.restype = ctypes.c_double
        c_img1 = (ctypes.c_double * len(img1))(*img1)
        c_img2 = (ctypes.c_double * len(img2))(*img2)
        choice = [1, 1, 0, 0, 0, 1, 0, 0, 0]
        choice = np.array(choice)
        choice = (ctypes.c_int * len(choice))(*choice)
        background = (ctypes.c_double)(0.0)
        return d2(c_img1, c_img2, ctypes.c_int(28), ctypes.c_int(28), choice, background)

    # KNN
    for idx in range(len(train_list)):
        time1 = time.time()
        model = KNN(twoSideTD)
        neigh, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
        eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], neigh)
        test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], neigh)
        time2 = time.time()
        with open('result record_2.txt', 'a') as f:
            f.write('kNN \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            result_record(f, train_acc, eval_acc, test_acc, neigh)


if __name__ == '__main__':
    main()
