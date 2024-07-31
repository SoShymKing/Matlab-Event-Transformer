function r = rotator(events)
   zeros([50,480,640,2]);
   res = [];
   for i = 1:length(events)
       tmp = events(:,i);
       res(tmp(1)+1,tmp(2)+1,tmp(3)+1,tmp(4)+1) = 1;
   end
    r = res;
end