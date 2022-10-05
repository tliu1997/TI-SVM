import numpy as np
import time
from Model import SVM, TISVM, RISVM, LOCSVM, LTISVM, LRISVM, TIRISVM, LTIRISVM, RIISVM, KNN, ResNetModel
from DataLoader import load_data, train_valid_split
import ctypes
import torch
import pytorch_lightning as pl
import torch.utils.data as data_utils
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score

# train_list = [100, 200, 500]
# val_list = [20, 40, 100]
# test_list = [20800, 20800, 20800]
train_list = [100]
val_list = [20]
test_list = [20800]


def result_record(file, train_acc, valid_acc, test_acc, svclassifier):
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

    # # TD-SVM
    # for idx in range(len(train_list)):
    #     time1 = time.time()
    #     model = TDSVM(twoSideTD)
    #     svclassifier, train_acc = model.train(x_train[0:train_list[idx], :], y_train[0:train_list[idx]])
    #     eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
    #     test_acc = model.evaluate(x_test[0:test_list[idx], :], y_test[0:test_list[idx]], svclassifier)
    #     time2 = time.time()
    #     with open('result record_2.txt', 'a') as f:
    #         f.write('TD-SVM \n')
    #         f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
    #         result_record(f, train_acc, eval_acc, test_acc, svclassifier)

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

    # KNN with tangent distance
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

    # ResNet (fill the gap of 200 training samples remained by best CNN)
    x_train = x_train.reshape((-1, 1, 28, 28))
    x_test = x_test.reshape((-1, 1, 28, 28))

    for idx in range(len(train_list)):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        train_ds = data_utils.TensorDataset(torch.from_numpy(x_train[0:train_list[idx], :]), torch.from_numpy(y_train[0:train_list[idx]]))
        test_ds = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        train_dl = data_utils.DataLoader(train_ds, batch_size=8, shuffle=True)
        test_dl = data_utils.DataLoader(test_ds, batch_size=8)
        time1 = time.time()
        model = ResNetModel()
        trainer = pl.Trainer(gpus=1, max_epochs=30)
        trainer.fit(model, train_dl)

        true_y, pred_y = [], []
        for batch in tqdm(iter(test_dl), total=len(test_dl)):
            x, y = batch
            true_y.extend(y)
            model.freeze()
            probs = torch.softmax(model(x), dim=1)
            preds = torch.argmax(probs, dim=1)
            pred_y.extend(preds.cpu())
        time2 = time.time()
        print(accuracy_score(true_y, pred_y))

        with open('result record.txt', 'a') as f:
            f.write('ResNet \n')
            f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
            f.write('Test: \n')
            f.write('accuracy: {:.3f}% \n'.format(accuracy_score(true_y, pred_y) * 100))
            f.write('\n')


if __name__ == '__main__':
    main()
