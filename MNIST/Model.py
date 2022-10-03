import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics import accuracy_score
from scipy import ndimage


"""This script defines the training, validation and testing process.
"""


# SVM with polynomial kernel
class SVM(object):
    def __init__(self, degree):
        self.degree = degree

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel='poly', degree=self.degree, coef0=1.0)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc


# SVM with translational-invariant kernel
class TISVM(object):
    def __init__(self, degree):
        self.degree = degree

    def ti_kernel(self, x, y):
        """
        Args:
            x: arrays of shape (n_samples1, n_features)
            y: arrays of shape (n_samples2, n_features)
        Returns:
            kernel_final: maximum kernel matrix of shape (n_samples1, n_samples2)
        """
        y_reshape = y.reshape((-1, 28, 28))

        # choose the maximum kernel for each pair of sample
        kernel_final = np.zeros((x.shape[0], y.shape[0]))

        for k in range(-5, 6):
            for l in range(-5, 6):
                # shift
                T_kl = np.roll(y_reshape, (k, l), axis=(1, 2))
                T_kl_reshape = T_kl.reshape((T_kl.shape[0], -1))

                kernel_medium = polynomial_kernel(x, T_kl_reshape, self.degree)
                kernel_final = np.maximum(kernel_medium, kernel_final)

        return kernel_final

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel=self.ti_kernel)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc


# SVM with translational-invariant kernel
class TIRISVM(object):
    def __init__(self, degree):
        self.degree = degree

    def ti_kernel(self, x, y):
        """
        Args:
            x: arrays of shape (n_samples1, n_features)
            y: arrays of shape (n_samples2, n_features)
        Returns:
            kernel_final: maximum kernel matrix of shape (n_samples1, n_samples2)
        """
        y_reshape = y.reshape((-1, 28, 28))

        # choose the maximum kernel for each pair of sample
        kernel_final = np.zeros((x.shape[0], y.shape[0]))

        for k in range(-7, 8):
            for l in range(-7, 8):
                for d in range(-10, 10):
                    # shift
                    T_kl = np.roll(y_reshape, (k, l), axis=(1, 2))
                    # rotate
                    T_rotate = ndimage.rotate(T_kl, d, axes=(1, 2), reshape=False)
                    T_kl_reshape = T_rotate.reshape((T_kl.shape[0], -1))

                    kernel_medium = polynomial_kernel(x, T_kl_reshape, self.degree)
                    kernel_final = np.maximum(kernel_medium, kernel_final)

        return kernel_final

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel=self.ti_kernel)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc


# SVM with translational-invariant kernel
class LTIRISVM(object):
    def __init__(self, degree, filter):
        self.degree = degree
        self.filter = filter

    def ti_kernel(self, x, y):
        """
        Args:
            x: arrays of shape (n_samples1, n_features)
            y: arrays of shape (n_samples2, n_features)
        Returns:
            kernel_final: maximum kernel matrix of shape (n_samples1, n_samples2)
        """
        y_reshape = y.reshape((-1, 28, 28))

        # choose the maximum kernel for each pair of sample
        kernel_final = np.zeros((x.shape[0], y.shape[0]))

        for k in range(-7, 8):
            for l in range(-7, 8):
                for d in range(-5, 6):
                    # rotate
                    T_rotate = ndimage.rotate(y_reshape, d, axes=(1, 2), reshape=False)
                    # shift
                    T_kl = np.roll(T_rotate, (k, l), axis=(1, 2))
                    T_kl_reshape = T_kl.reshape((T_kl.shape[0], -1))

                    kernel_medium = self.locality_kernel(x, T_kl_reshape)
                    kernel_final = np.maximum(kernel_medium, kernel_final)

        return kernel_final

    # using im2col to accelerate computation (memory locality)
    def im2col(self, x, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = x.shape

        assert (H + 2 * pad - filter_h) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
        assert (W + 2 * pad - filter_w) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))

        for y in range(out_h):
            for x in range(out_w):
                col[:, :, y, x, :, :] = img[:, :, y * stride:y * stride + filter_h, x * stride:x * stride + filter_w]

        col = col.transpose(0, 2, 3, 4, 5, 1).reshape((N, -1))
        return col

    def locality_kernel(self, x, y):
        X1 = np.transpose(x.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))
        Y1 = np.transpose(y.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))

        Z = np.zeros((x.shape[0], y.shape[0]))

        # using im2col to accelerate computation (memory locality)
        filter_h = self.filter
        filter_w = self.filter
        stride = 1
        pad = (self.filter - 1) // 2
        X2 = self.im2col(X1, filter_h, filter_w, stride, pad)
        Y2 = self.im2col(Y1, filter_h, filter_w, stride, pad)

        # Hyperparameter: Degree
        D1 = 2
        D2 = self.degree // D1

        for i in range(0, X2.shape[1], filter_h * filter_w):
            Z = Z + polynomial_kernel(X2[:, i: i + filter_h * filter_w], Y2[:, i: i + filter_h * filter_w],
                                      degree=D1, coef0=1.0)

        Z = (1 / (X1.shape[1] * X1.shape[2]) * Z + np.ones((x.shape[0], y.shape[0]))) ** D2
        return Z

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel=self.ti_kernel)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc


# SVM with translational-invariant kernel
class LTISVM(object):
    def __init__(self, degree, filter):
        self.degree = degree
        self.filter = filter

    def ti_kernel(self, x, y):
        """
        Args:
            x: arrays of shape (n_samples1, n_features)
            y: arrays of shape (n_samples2, n_features)
        Returns:
            kernel_final: maximum kernel matrix of shape (n_samples1, n_samples2)
        """
        y_reshape = y.reshape((-1, 28, 28))

        # choose the maximum kernel for each pair of sample
        kernel_final = np.zeros((x.shape[0], y.shape[0]))

        for k in range(-7, 8):
            for l in range(-7, 8):
                # shift
                T_kl = np.roll(y_reshape, (k, l), axis=(1, 2))
                T_kl_reshape = T_kl.reshape((T_kl.shape[0], -1))

                kernel_medium = self.locality_kernel(x, T_kl_reshape)
                kernel_final = np.maximum(kernel_medium, kernel_final)

        return kernel_final

    # using im2col to accelerate computation (memory locality)
    def im2col(self, x, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = x.shape

        assert (H + 2 * pad - filter_h) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
        assert (W + 2 * pad - filter_w) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))

        for y in range(out_h):
            for x in range(out_w):
                col[:, :, y, x, :, :] = img[:, :, y * stride:y * stride + filter_h, x * stride:x * stride + filter_w]

        col = col.transpose(0, 2, 3, 4, 5, 1).reshape((N, -1))
        return col

    def locality_kernel(self, x, y):
        X1 = np.transpose(x.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))
        Y1 = np.transpose(y.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))

        Z = np.zeros((x.shape[0], y.shape[0]))

        # using im2col to accelerate computation (memory locality)
        filter_h = self.filter
        filter_w = self.filter
        stride = 1
        pad = (self.filter - 1) // 2
        X2 = self.im2col(X1, filter_h, filter_w, stride, pad)
        Y2 = self.im2col(Y1, filter_h, filter_w, stride, pad)

        # Hyperparameter: Degree
        D1 = 2
        D2 = self.degree // D1

        for i in range(0, X2.shape[1], filter_h * filter_w):
            Z = Z + polynomial_kernel(X2[:, i: i + filter_h * filter_w], Y2[:, i: i + filter_h * filter_w],
                                      degree=D1, coef0=1.0)

        Z = (1 / (X1.shape[1] * X1.shape[2]) * Z + np.ones((x.shape[0], y.shape[0]))) ** D2
        return Z

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel=self.ti_kernel)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc


# SVM with local connective kernel
class LOCSVM(object):
    def __init__(self, degree, filter):
        self.degree = degree
        self.filter = filter

    # using im2col to accelerate computation (memory locality)
    def im2col(self, x, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = x.shape

        assert (H + 2 * pad - filter_h) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
        assert (W + 2 * pad - filter_w) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))

        for y in range(out_h):
            for x in range(out_w):
                col[:, :, y, x, :, :] = img[:, :, y * stride:y * stride + filter_h, x * stride:x * stride + filter_w]

        col = col.transpose(0, 2, 3, 4, 5, 1).reshape((N, -1))
        return col

    def locality_kernel(self, x, y):
        X1 = np.transpose(x.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))
        Y1 = np.transpose(y.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))

        Z = np.zeros((x.shape[0], y.shape[0]))

        # using im2col to accelerate computation (memory locality)
        filter_h = self.filter
        filter_w = self.filter
        stride = 1
        pad = (self.filter - 1) // 2
        X2 = self.im2col(X1, filter_h, filter_w, stride, pad)
        Y2 = self.im2col(Y1, filter_h, filter_w, stride, pad)

        # Hyperparameter: Degree
        D1 = 2
        D2 = self.degree // D1

        for i in range(0, X2.shape[1], filter_h * filter_w):
            Z = Z + polynomial_kernel(X2[:, i: i + filter_h * filter_w], Y2[:, i: i + filter_h * filter_w],
                                      degree=D1, coef0=1.0)

        Z = (1 / (X1.shape[1] * X1.shape[2]) * Z + np.ones((x.shape[0], y.shape[0]))) ** D2
        return Z

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel=self.locality_kernel)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc


# SVM with rotation-invariant kernel
class RISVM(object):
    def __init__(self, degree):
        self.degree = degree

    def ri_kernel(self, x, y):
        """
        Args:
            x: arrays of shape (n_samples1, n_features)
            y: arrays of shape (n_samples2, n_features)
        Returns:
            kernel_final: maximum kernel matrix of shape (n_samples1, n_samples2)
        """
        y_reshape = y.reshape((-1, 28, 28))

        # choose the maximum kernel for each pair of sample
        kernel_final = np.zeros((x.shape[0], y.shape[0]))

        for d in range(-10, 10):
            # rotate
            T_rotate = ndimage.rotate(y_reshape, d, axes=(1, 2), reshape=False)
            T_rotate_reshape = T_rotate.reshape((T_rotate.shape[0], -1))

            kernel_medium = polynomial_kernel(x, T_rotate_reshape, self.degree)
            kernel_final = np.maximum(kernel_medium, kernel_final)

        return kernel_final

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel=self.ri_kernel)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc


# SVM with rotation-invariant kernel
class RIISVM(object):
    def __init__(self, degree):
        self.degree = degree

    def ri_kernel(self, x, y):
        """
        Args:
            x: arrays of shape (n_samples1, n_features)
            y: arrays of shape (n_samples2, n_features)
        Returns:
            kernel_final: maximum kernel matrix of shape (n_samples1, n_samples2)
        """
        y_reshape = y.reshape((-1, 28, 28))

        # choose the maximum kernel for each pair of sample
        kernel_final = np.zeros((x.shape[0], y.shape[0]))

        for d in range(-10, 10):
            # rotate
            T_rotate = ndimage.rotate(y_reshape, d, axes=(1, 2), reshape=False)
            T_rotate_reshape = T_rotate.reshape((T_rotate.shape[0], -1))

            kernel_final = kernel_final + polynomial_kernel(x, T_rotate_reshape, self.degree)

        return kernel_final/20

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel=self.ri_kernel)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc


# SVM with rotation-invariant kernel
class LRISVM(object):
    def __init__(self, degree, filter):
        self.degree = degree
        self.filter = filter

    def ri_kernel(self, x, y):
        """
        Args:
            x: arrays of shape (n_samples1, n_features)
            y: arrays of shape (n_samples2, n_features)
        Returns:
            kernel_final: maximum kernel matrix of shape (n_samples1, n_samples2)
        """
        y_reshape = y.reshape((-1, 28, 28))

        # choose the maximum kernel for each pair of sample
        kernel_final = np.zeros((x.shape[0], y.shape[0]))

        for d in range(-5, 6):
            # rotate
            T_rotate = ndimage.rotate(y_reshape, d, axes=(1, 2), reshape=False)
            T_rotate_reshape = T_rotate.reshape((T_rotate.shape[0], -1))

            kernel_medium = self.locality_kernel(x, T_rotate_reshape)
            kernel_final = np.maximum(kernel_medium, kernel_final)

        return kernel_final

    # using im2col to accelerate computation (memory locality)
    def im2col(self, x, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = x.shape

        assert (H + 2 * pad - filter_h) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
        assert (W + 2 * pad - filter_w) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))

        for y in range(out_h):
            for x in range(out_w):
                col[:, :, y, x, :, :] = img[:, :, y * stride:y * stride + filter_h, x * stride:x * stride + filter_w]

        col = col.transpose(0, 2, 3, 4, 5, 1).reshape((N, -1))
        return col

    def locality_kernel(self, x, y):
        X1 = np.transpose(x.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))
        Y1 = np.transpose(y.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))

        Z = np.zeros((x.shape[0], y.shape[0]))

        # using im2col to accelerate computation (memory locality)
        filter_h = self.filter
        filter_w = self.filter
        stride = 1
        pad = (self.filter - 1) // 2
        X2 = self.im2col(X1, filter_h, filter_w, stride, pad)
        Y2 = self.im2col(Y1, filter_h, filter_w, stride, pad)

        # Hyperparameter: Degree
        D1 = 2
        D2 = self.degree // D1

        for i in range(0, X2.shape[1], filter_h * filter_w):
            Z = Z + polynomial_kernel(X2[:, i: i + filter_h * filter_w], Y2[:, i: i + filter_h * filter_w],
                                      degree=D1, coef0=1.0)

        Z = (1 / (X1.shape[1] * X1.shape[2]) * Z + np.ones((x.shape[0], y.shape[0]))) ** D2

        return Z

    def train(self, x, y):
        svclassifier = SVC(gamma='auto', kernel=self.ri_kernel)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc