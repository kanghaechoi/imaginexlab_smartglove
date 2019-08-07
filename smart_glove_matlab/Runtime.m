codeInitialize;
path = "\dairy\";
MAXFEATURES = 26;
NUMBEROFITERATION = 5;
for NumberFeature = 1:26
    for loop = 1:NUMBEROFITERATION
        DairyName = strcat('LSTM_', int2str(NumberFeature),"_", int2str(loop), '.log');
        f = fullfile('dairy',DairyName);
        fid =  fopen( f, 'w' );
        fclose(fid);
        diary(f);
        smartgloveRuntime(3,NumberFeature)
    end
end