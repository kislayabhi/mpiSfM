#include "mpiSfM/SIFT++.hpp"
#include <string>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/filesystem.hpp>

using namespace std;
namespace mpi = boost::mpi;

void load_images(string image_filepath, vector<string>& image_filenames);

int main(int argc, char** argv)
{
	mpi::environment env;
	mpi::communicator world;

	if (world.rank() == 0) {

		string image_filepath = "data/sample_images/";
		string key_filepath = "data/generated_keys/";
		vector<string> image_filenames;
		load_images(image_filepath, image_filenames);

		int size = world.size();

		vector<vector<string> > images_per_chunk(size - 1);
		int chunk_no = 0;
		for (int i = 0; i < image_filenames.size(); i++, chunk_no++) {
			if (image_filenames[i].substr(image_filenames[i].size() - 4, 4) == ".bmp") {
				chunk_no = chunk_no % (size - 1);
				images_per_chunk[chunk_no].push_back(image_filenames[i]);
			}
		}
		for (int i = 1; i < size; i++) {
			world.send(i, 0, image_filepath);
			world.send(i, 0, key_filepath);
			world.send(i, 0, images_per_chunk[i - 1]);
		}

	}else {
		vector<string> image_filenames;
		string image_filepath, key_filepath;

		world.recv(0, 0, image_filepath);
		world.recv(0, 0, key_filepath);
		world.recv(0, 0, image_filenames);

		cout << "Rank: " << world.rank() << endl;

		for (int pc = 0; pc < image_filenames.size(); pc++)
			// generate_keypoints(image_filenames[pc], image_filenames[pc].substr(0, image_filenames[pc].size() - 4) + ".key");
			generate_keypoints(image_filepath + image_filenames[pc], key_filepath + image_filenames[pc].substr(0, image_filenames[pc].size() - 4) + ".key");
	}
}


/* Loads image names as strings from a directory*/
void load_images(string image_filepath, vector<string>& image_filenames)
{
	boost::filesystem::path p(image_filepath.c_str());

	try
	{
		if (exists(p)) {
			if (is_directory(p)) {
				cout << p << " Directory Containing Images Found:\n";
				for (boost::filesystem::directory_entry& x : boost::filesystem::directory_iterator(p))
					image_filenames.push_back(x.path().filename().string());
			}else
				cout << p << " exists, but is not a directory\n";
		}else
			cout << p << " does not exist\n";
	}
	catch (const boost::filesystem::filesystem_error& ex)
	{
		cout << ex.what() << '\n';
	}
}
