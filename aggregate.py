import zarr 
import numpy as np

class Aggregate:
    def __init__(self, path) -> None:
        """
            TODO: implement the usage of flags. for example, 
            when storing a dataset which is not aggregated 
            set the flag to false for those specific chunks.
            So when aggregating it has to compare the flags and 
            chunks with the history in order to aggregate.
        
        """
        self.path = path
        self.root = zarr.open_group(store=path, mode = "r+")    

    def _chunk(self):
        # TODO: Maybe define a function which finds an optimal chunk for aggregation
        count = self.root["_build"]["count"][:]

        return count
    
    def get_correct_count(self, root):
        # TODO : Very bad way of updating max, using all arrays and not last listed and checks against current

        _count = root["count"][:]
        if len(_count) == 1:
            root["count"] = _count
        
        self.root["count"] = _count.sum(axis = 0)
        #return _count.sum(axis = 0)

    def stdev(self, root):
        N = root["count"][:]
        sqrs = root["squares"][:]
        mns = root["mean"][:]
        variance = (sqrs - N*mns**2)/(N - 1) 
        _stdev = np.sqrt(variance)

        root["stdev"] = _stdev



    def means(self, root):
        #root = zarr.open_group(store=self.path, mode = "r+")
        N = root["count"][:]
        _sums = root["sums"]
        _mean = _sums/N 
        root["mean"] = _mean

        #return None

    def __update_max(self, root):
        # TODO : Very bad way of updating max, using all arrays and not last listed and checks against current
        max_root = root["maximum"][:]
        if len(max_root) == 1:
            self.root["maximum"] = max_root
   
        self.root["maximum"] = max_root.max(axis = 0)
        

    def __update_min(self, root):
        # TODO : Very bad way of updating max, using all arrays and not last listed and checks against current

        min_root = root["minimum"][:]
        if len(min_root) == 1:
            self.root["minimum"] = min_root
   
        self.root["minimum"] = min_root.min(axis = 0)

    def __update_sums(self, root):
        # TODO : Very bad way of updating max, using all arrays and not last listed and checks against current

        sum_root = root["sums"][:]
        if len(sum_root) == 1:
            self.root["sums"] = sum_root
   
        self.root["sums"] = sum_root.sum(axis = 0)

    def __update_squares(self, root):
        # TODO : Very bad way of updating max, using all arrays and not last listed and checks against current

        squares_root = root["squares"][:]
        if len(squares_root) == 1:
            self.root["squares"] = squares_root
   
        self.root["squares"] = squares_root.sum(axis = 0) 

    @property
    def update_metadata(self):
        from zarrregistry import ZarrRegistery
        # TODO : implement register to update metadata
        return ZarrRegistery(path=self.path)
    

    def update_statistics(self):
        # TODO : rename this method to something more appropiate, misleading atm

        print("Using _build to aggregate statistics.")

        _build = self.root["_build"]
        # print("build store", _build.shape)

        self.__update_max(root = _build)
        self.__update_min(root =_build)
        self.__update_sums(root = _build)
        self.__update_squares(root = _build)
        

        
        self.get_correct_count(root = _build)

        self.means(root = self.root)
        self.stdev(root = self.root)

        print("Data is aggregated and placed in root statistics folder")
        self.update_metadata.update_history(action= "Statistics aggregated from _build and added to root folder.")
        