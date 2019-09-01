import sys
import os 
class Logger():

    def __init__(self, parent_folder, save_file):

        #self.in_build_classes = ['int', 'float', 'bool', 'numpy.ndarray', 'list', 'dict', 'tuple',
        #							'str']
        self.orig_stdout = sys.stdout
        self.save_file = parent_folder+'/'+save_file
        if os.path.exists(parent_folder):
            pass
        else:
            os.makedirs(parent_folder)

        self.black_list = ['annotation_dict',
                           'annotation_list',
                           'pedestrian_dict']
        #self.file_to_write = open(save_file,'w')

		#sys.stdout = self.file_to_write


    def log_info(self, information_dict):
        with open(self.save_file, 'a') as self.file_to_write:
            for key in information_dict:
                if key not in self.black_list:
                    print("%s - %s" % (key, information_dict[key]), file=self.file_to_write)
        self.file_to_write.close()


    def log(self, text):
        with open(self.save_file, 'a') as self.file_to_write:
            print(text, file=self.file_to_write)
        self.file_to_write.close()


    def log_header(self, text):

        l = len(text)
        h_line = ''
        for i in range(len(text)):
            h_line+='#'
        with open(self.save_file, 'a') as self.file_to_write:
            print("\n\n")
            print(h_line, file=self.file_to_write)
            print(text, file=self.file_to_write)
            print(h_line, file=self.file_to_write)
        self.file_to_write.close()

    def close_logger(self):

        self.file_to_write.close()


if __name__=='__main__':

    test_dict = {'a':10, 'b':'sfd'}
    log = Logger('./saving_file.txt')
    print('save this')
    print('and this')
    log.log_info(test_dict)
    log.close_logger()



