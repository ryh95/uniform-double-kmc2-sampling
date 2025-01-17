data_names = ["a2","a3","b2-random-10","b2-random-15","b2-random-20"];
% data_names = ["bio_train","rna","poker-hand"];
algorithms = ["kmc2","uniform-wo-wk","Double-kmc2"];
% algorithms = ["kmc2"];
num_points = [5250,7500,10000,15000,20000];
% num_points = [145751,488565,1000000];
data_type = "sum-squared-distances";
field_type = 'sum_squared_distances';
% data_type = "running-time";
% field_type = 'running_time';
data = zeros(length(algorithms),length(data_names));
for i = 1:length(data_names)
    for j = 1:length(algorithms)
        v_struct = load(join([data_names(i),algorithms(j),data_type,'.mat'],"_"));
        value = getfield(v_struct,field_type);
        data(j,i) = mean(value);
    end
end
is_bar = true;
f = figure();
if is_bar
    data_names(1) = "KDD";
    data_names(2) = "RNA";
    data_names(3) = "Poker Hand";
    c_data_names = categorical(data_names);
    c_data_names = reordercats(c_data_names,cellstr(data_names));
    bar(c_data_names,data');
    legend('kmc2','uni-w/o-replace','Double-kmc2','Location','best');
else
    p = plot(num_points,data','LineWidth',1,'MarkerSize',4);
    set(p,{'color'},{'b';'k';'r'});
    set(p,{'Marker'},{'o';'s';'v'});
    set(p,{'MarkerFaceColor'},{'b';'k';'r'});


    xlabel('number of points');

    if strcmp(data_type,"running-time")
        ylabel('number of distance evaluations');
    else
        ylabel('sum of squared distances');
    end
    legend('kmc2','uni-w/o-replace','Double-kmc2','Location','northwest');
end

% crop figure borderline and change config
% ref: https://www.mathworks.com/matlabcentral/answers/471281-save-plot-with-minimal-white-space-crop-padding
% ref: https://www.mathworks.com/help/matlab/creating_plots/save-figure-with-minimal-white-space.html
% my config: crop,font_size 15,font_type dejavu sans
% saveas(f,strcat(data_type,'.jpg'));