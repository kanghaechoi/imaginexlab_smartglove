function readFolders()
    folderPath = "/Sorted_Data/";
    subFolderPath = ["20s/","40s/","60s/"];
    genderPath = ["Male/","Female/"];
    
     for sP = 1 : size(subFolderPath,2)
         for gP = 1 : size(genderPath,2)
            path = folderPath + subFolderPath(sP) + genderPath(gP);
            fprintf("%s\n",path)
         end
     end
   
end


