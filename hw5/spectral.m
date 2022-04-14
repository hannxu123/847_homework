function [label, step] = spectral(data, clusters, dim)
   [num, d] = size(data);
   index = randperm(num, clusters);
   dis = zeros(num, clusters);
   label = zeros(num, 1);
   % center = data(index, :);

   step = 0;

   [evectors, evalues] = eig(data * (data.'));
   data = evectors(:, 1:(1+dim-1));
   center = data(index, :) 

   while(1) 
       % save the previous step centers
       pre_center = center;

       % calculate distance between samples and centers
       for i = 1:num
           for j = 1:clusters
               dis(i, j) = norm(data(i,:) - center(j, :));
           end
       end

       % construct new clutters
       for i = 1:num
           label(i) = find(dis(i,:)==min(dis(i,:)));
       end

       % attain new centers
       for i = 1:clusters
           one_clutter = data(find(label==i), :);
           center(i, :) = mean(one_clutter);
       end

       % check whether there is new updates
       if (center == pre_center)
           break;
       end
       step = step + 1;
   end
end
