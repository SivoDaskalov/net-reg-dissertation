
outfilename='summary.txt';
outfile=fopen(outfilename,'a');

for big=0:1,
    if (big==0)
        name='small';
        nTF=3;
        p = 33;
    else
        name='big';
        nTF=10;
        p=110;
    end
    for num=1:5,
        fprintf(outfile, 'Setup=%d p=%d\n',num, p);
        infilename = strcat('setup-',name,int2str(num),'.txt');
        data = dlmread(infilename,',',0,0);
        m1 = mean(data,1);
				m2 = median(data, 1);
        s = std(data,0,1);
        % step1 prediction
        fprintf(outfile,'step1 prediction\n');
        for i=2:5,
            fprintf(outfile, '%5.2f(%5.2f)[%5.2f] ', m1(i), s(i), m2(i));
        end
        fprintf(outfile,'\n');
        % step2 prediction
        fprintf(outfile,'step2 prediction\n');
        for i=2:5,
            fprintf(outfile, '%5.2f(%5.2f)[%5.2f] ', m1(5+nTF*11+i), s(5+nTF*11+i),m2(5+nTF*11+i));
        end
        fprintf(outfile,'\n');
        % step1 solution
        fprintf(outfile,'step1 solution\n');
        for i=1:nTF,
            for j=1:11,
                fprintf(outfile, '%5.2f(%5.2f)[%5.2f] ', m1(5+(i-1)*11+j), s(5+(i-1)*11+j), m2(5+(i-1)*11+j));
            end
            fprintf(outfile, '\n');
        end
        % step2 solution
        fprintf(outfile,'step2 solution\n');
        for i=1:nTF,
            for j=1:11,
                fprintf(outfile, '%5.2f(%5.2f)[%5.2f] ', m1(10+nTF*11+(i-1)*11+j), s(10+nTF*11+(i-1)*11+j), m2(10+nTF*11+(i-1)*11+j));
            end
            fprintf(outfile, '\n');
        end
        fprintf(outfile,'\n');
    end
end
fclose(outfile);
