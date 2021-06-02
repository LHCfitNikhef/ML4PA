from ncempy.io import dm
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
#from matplotlib import rc
import trainZLP
import torch.utils.data as data

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': 15})
#rc('text', usetex=True)


class Spectral_image(data.Dataset):

    def __init__(self, data, deltadeltaE, pixelsize=None, name=None):
        super().__init__()
        self.data = data
        self.ddeltaE = deltadeltaE
        self.deltaE = self.determine_deltaE()
        self.x_axis, self.y_axis = self.calc_axes()
        self.data_zoomed = None
        self.data_zoomed_concat = None
        self.data_unc = None
        if pixelsize is not None:
            self.pixelsize = pixelsize * 1E6

    def calc_axes(self):
        """
        Calculates the attribustes x_axis and y_axis of the image. These are the spatial axes, and \
            can be used to find the spatial location of a pixel and are used in the plotting functions.

        If one wants to alter these axis, one can do this manually by running image.x_axis = ..., \
            and image.y_axis = ....

        Returns
        -------
        None.

        """
        y_axis = np.linspace(0, self.image_shape[0] - 1, self.image_shape[0])
        x_axis = np.linspace(0, self.image_shape[1] - 1, self.image_shape[1])
        if hasattr(self, 'pixelsize'):
            y_axis *= self.pixelsize[0]
            x_axis *= self.pixelsize[1]
        return x_axis, y_axis

    def determine_deltaE(self):
        data_avg = np.average(self.data, axis=(0, 1))
        ind_max = np.argmax(data_avg)
        deltaE = np.linspace(-ind_max * self.ddeltaE, (self.l - ind_max - 1) * self.ddeltaE, self.l)
        return deltaE


    def get_ticks(self, sig=2, n_tick=10):
        """
        Generates ticks of (spatial) x- and y axis for plotting perposes.

        Parameters
        ----------
        sig : int, optional
            Scientific signifance of ticsk. The default is 2.
        n_tick : int, optional
            desired number of ticks. The default is 10.

        Returns
        -------
        xlabels : np.array of type object
        ylabels : np.array of type object

        """
        fmt = '%.' + str(sig) + 'g'
        xlabels = np.zeros(self.x_axis.shape, dtype=object)
        xlabels[:] = ""
        each_n_pixels = math.floor(len(xlabels) / n_tick)
        for i in range(len(xlabels)):
            if i % each_n_pixels == 0:
                xlabels[i] = '%s' % float(fmt % self.x_axis[i])
        ylabels = np.zeros(self.y_axis.shape, dtype=object)
        ylabels[:] = ""
        each_n_pixels = math.floor(len(ylabels) / n_tick)
        for i in range(len(ylabels)):
            if i % each_n_pixels == 0:
                ylabels[i] = '%s' % float(fmt % self.y_axis[i])
        return xlabels, ylabels

    def show_image(self, title=None, xlab=None, ylab=None, pixel_highlight=None, selection=np.array([[0, 10], [0, 10]])):
        """
        INPUT:
            self -- spectral image
            title -- str, delfault = None, title of plot
            xlab -- str, default = None, x-label
            ylab -- str, default = None, y-label
        OUTPUT:
        Plots the summation over the intensity for each pixel in a heatmap.
        """
        # TODO: invert colours
        if hasattr(self, 'name'):
            name = self.name
        else:
            name = ''
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        if title is None:
            plt.title('Integrated intensity spectrum')
        else:
            plt.title(title)
        if hasattr(self, 'pixelsize'):
            xticks, yticks = self.get_ticks(sig=0) #TODO signifance does not seem to change digits
            sns.heatmap(np.sum(self.data, axis=2), xticklabels=xticks, yticklabels=yticks)
            plt.xlabel('micron')
            plt.ylabel('micron')
        else:
            sns.heatmap(np.sum(self.data, axis=2))
        if xlab is not None:
            plt.xlabel(xlab)
        if ylab is not None:
            plt.ylabel(ylab)

        if pixel_highlight is not None:
            rect = plt.Rectangle(pixel_highlight, 1, 1, color="C0", linewidth=3, fill=False, clip_on=False)
            ax.add_patch(rect)

        if selection is None:
            plt.show()
        else:
            window_xmin = selection[0, 0]
            window_width = selection[0, 1]
            window_ymin = selection[1, 0]
            window_height = selection[1, 1]
            # full_width, full_height = self.image_shape[1], self.image_shape[0]
            # window_xmin = int(selection[0,0] * full_width)
            # window_width = int(selection[0,1] * full_width)
            # window_ymin = int(selection[1,0] * full_height)
            # window_height = int(selection[1,1] * full_height)
            rect = plt.Rectangle([window_xmin, window_ymin], window_width, window_height, color="gold",
                                 linewidth=3, fill=False, clip_on=False)
            ax.add_patch(rect)
            plt.show()

            # zoomed in plot
            xticks_zoom, yticks_zoom = None, None  # TODO: add custom x and y ticks for the zoomed in plot
            self.data_zoomed = self.data[window_ymin: window_ymin + window_height, window_xmin: window_xmin + window_width, :]
            sns.heatmap(np.sum(self.data_zoomed,axis=2))
            plt.title("Integrated intensity spectrum close up " + name)
            plt.show()

        self.data_zoomed_concat = np.reshape(self.data_zoomed, (-1, self.l))
        epsilon = 1e-3
        self.data_zoomed_concat[self.data_zoomed_concat < 0] = epsilon
        ci_low = np.nanpercentile(np.log(self.data_zoomed_concat), 16, axis=0)
        ci_high = np.nanpercentile(np.log(self.data_zoomed_concat), 84, axis=0)
        self.data_unc= np.absolute(ci_high - ci_low)



    def plot_spectrum(self, i, j, normalize=False, signal="EELS", log=False):
        signal_pixel = self.get_pixel_signal(i, j, signal)
        if normalize:
            signal_pixel /= np.max(np.absolute(signal_pixel))
        if log:
            signal_pixel = np.log(signal_pixel)
            plt.ylabel("log intensity")
        plt.plot(self.deltaE, signal_pixel, label="[" + str(j) + "," + str(i) + "]")
        plt.legend()
        plt.show()
        return signal_pixel

    def get_pixel_signal(self, i, j, signal='EELS'):
        """
        INPUT:
            i: int, x-coordinate for the pixel
            j: int, y-coordinate for the pixel
        Keyword argument:
            signal: str (default = 'EELS'), what signal is requested, should comply with defined names
        OUTPUT:
            signal: 1D numpy array, array with the requested signal from the requested pixel
        """
        return self.data[i, j, :]

    def train_ZLPs(self, training_data, **kwargs):
        trainZLP.train_nn_scaled(self, training_data, **kwargs)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data_zoomed_concat[idx]
        return self.deltaE[idx], data_point, self.data_unc[idx]

    def __len__(self):
        return self.l

    @property
    def l(self):
        """returns length of spectra, i.e. num energy loss bins"""
        return self.data.shape[2]

    @property
    def image_shape(self):
        """return 2D-shape of spectral image"""
        return self.data.shape[:2]

    @staticmethod
    def get_prefix(unit, SIunit=None, numeric=True):
        if SIunit is not None:
            lenSI = len(SIunit)
            if unit[-lenSI:] == SIunit:
                prefix = unit[:-lenSI]
                if len(prefix) == 0:
                    if numeric:
                        return 1
                    else:
                        return prefix
            else:
                print("provided unit not same as target unit: " + unit + ", and " + SIunit)
                if numeric:
                    return 1
                else:
                    return prefix
        else:
            prefix = unit[0]
        if not numeric:
            return prefix

        if prefix == 'p':
            return 1E-12
        if prefix == 'n':
            return 1E-9
        if prefix in ['μ', 'µ', 'u', 'micron']:
            return 1E-6
        if prefix == 'm':
            return 1E-3
        if prefix == 'k':
            return 1E3
        if prefix == 'M':
            return 1E6
        if prefix == 'G':
            return 1E9
        if prefix == 'T':
            return 1E12
        else:
            print("either no or unknown prefix in unit: " + unit + ", found prefix " + prefix + ", asuming no.")
        return 1

    @classmethod
    def load_data(cls, path_to_dmfile, load_additional_data=False):
        """
        INPUT:
            path_to_dmfile: str, path to spectral image file (.dm3 or .dm4 extension)
        OUTPUT:
            image -- Spectral_image, object of Spectral_image class containing the data of the dm-file
        """
        dmfile_tot = dm.fileDM(path_to_dmfile)
        additional_data = []
        for i in range(dmfile_tot.numObjects - dmfile_tot.thumbnail * 1):
            dmfile = dmfile_tot.getDataset(i)
            if dmfile['data'].ndim == 3:
                dmfile = dmfile_tot.getDataset(i)
                data = np.swapaxes(np.swapaxes(dmfile['data'], 0, 1), 1, 2)
                if not load_additional_data:
                    break
            elif load_additional_data:
                additional_data.append(dmfile_tot.getDataset(i))
            if i == dmfile_tot.numObjects - dmfile_tot.thumbnail * 1 - 1:
                if (len(additional_data) == i + 1) or not load_additional_data:
                    print("No spectral image detected")
                    dmfile = dmfile_tot.getDataset(0)
                    data = dmfile['data']

        ddeltaE = dmfile['pixelSize'][0]
        pixelsize = np.array(dmfile['pixelSize'][1:])
        energyUnit = dmfile['pixelUnit'][0]
        ddeltaE *= cls.get_prefix(energyUnit, 'eV')
        pixelUnit = dmfile['pixelUnit'][1]
        pixelsize *= cls.get_prefix(pixelUnit, 'm')
        image = cls(data, ddeltaE, pixelsize=pixelsize, name=path_to_dmfile[:-4])
        if load_additional_data:
            image.additional_data = additional_data
        return image
