data_names = ["3","kitten","2"];
algorithms = ["uni-sample","kmc2","Double-kmc2"];
num_points = [900,3600,14400];
data_type = "running_time"; % running_time/obj
% data_type = "obj";
data = zeros(length(algorithms),length(data_names));
for i = 1:length(data_names)
    for j = 1:length(algorithms)
        nr = num2str(sqrt(num_points(i)));
        v_struct = load(join([algorithms(j),data_names(i),nr,nr,'.mat'],"_"));
        value = getfield(v_struct,data_type);
        data(j,i) = mean(value);
    end
end
f = figure('visible','off');
% data' because it has the same first dim with num_points
p = plot(num_points,data','LineWidth',1,'MarkerSize',4);
set(p,{'color'},{'b';'k';'r'});
set(p,{'Marker'},{'o';'s';'v'});
set(p,{'MarkerFaceColor'},{'b';'k';'r'});

legend('uni-w/o-replace','kmc2','Double-kmc2','Location','best');
xlabel('number of points');
if strcmp(data_type,"running_time")
    ylabel('number of distance evaluations');
elseif strcmp(data_type,"obj")
    ylabel('sum of squared distances');
end
saveas(f,strcat(data_type,'.jpg')); % image-running-time/ncut