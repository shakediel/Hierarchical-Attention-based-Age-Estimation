import h5py


class DataParser:
	def __init__(self, hdf5_file_path):
		self.hdf5_file_path = hdf5_file_path
		self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

	def initialize_data(self):
		self.x_train, self.y_train, self.x_test, self.y_test = self.read_data_from_h5_file(self.hdf5_file_path)

	@staticmethod
	def read_data_from_h5_file(hdf5_path):
		hdf5_file = h5py.File(hdf5_path, "r")

		return hdf5_file["train_img"][:], hdf5_file["train_labels"][:], \
		       hdf5_file["test_img"][:], hdf5_file["test_labels"][:]
