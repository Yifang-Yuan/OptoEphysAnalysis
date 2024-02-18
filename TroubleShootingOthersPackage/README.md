**Trouble 1. Recording folders are read in alphabetical order: 1, 10, 11, 2,3...if you have more than 10 recordings**

Replace `BinaryRecording.py` in this package, https://github.com/open-ephys/open-ephys-python-tools.

Another option is to replace the following function in the orginal package.

It uses `enumerate` to read recording* folders, which is in an order of : 1,10,2,3,...


    def detect_recordings(directory):
        
        recordings = []
        experiment_directories = glob.glob(os.path.join(directory, 'experiment*'))
        experiment_directories.sort()

        for experiment_index, experiment_directory in enumerate(experiment_directories):
             
            recording_directories = glob.glob(os.path.join(experiment_directory, 'recording*'))
            recording_directories.sort()
            print (experiment_index,'---')
            
            for recording_index, recording_directory in enumerate(recording_directories):
            
                recordings.append(BinaryRecording(recording_directory, 
                                                        experiment_index,
                                                        recording_index))
                print (recording_index)
                print (recording_directory)
                
        return recordings


I changed this function to make sure folders are read in numeric order.


    def detect_recordings(directory):

        recordings = []
        # Custom sorting key function
        def numeric_sort_key_exp(recording_directory):
            # Split the directory name based on the prefix "recording"
            prefix, suffix = recording_directory.split('experiment')
            try:
                # Convert the remaining part to an integer for sorting
                numeric_suffix = int(suffix)
            except ValueError:
                # If the suffix is not numeric, return a high value to ensure it comes later in sorting
                numeric_suffix = float('inf')
            return numeric_suffix
        experiment_directories = glob.glob(os.path.join(directory, 'experiment*'))
        experiment_directories.sort(key=numeric_sort_key_exp)  # Sort experiment directories numerically
        
        for experiment_index, experiment_directory in enumerate(experiment_directories):
            print(experiment_index, '---')
        # Custom sorting key function for recording directories
            def numeric_sort_key_recording(recording_directory):
                # Split the directory name based on the prefix "recording"
                prefix, suffix = recording_directory.split('recording')
                try:
                    # Convert the remaining part to an integer for sorting
                    numeric_suffix = int(suffix)
                except ValueError:
                    # If the suffix is not numeric, return a high value to ensure it comes later in sorting
                    numeric_suffix = float('inf')
                return numeric_suffix
            recording_directories = glob.glob(os.path.join(experiment_directory, 'recording*'))
            recording_directories.sort(key=numeric_sort_key_recording)  # Sort recording directories numerically
            for recording_index, recording_directory in enumerate(recording_directories):
                recordings.append(BinaryRecording(recording_directory, experiment_index, recording_index))
                print(recording_index)
                print(recording_directory)    
                
        return recordings


