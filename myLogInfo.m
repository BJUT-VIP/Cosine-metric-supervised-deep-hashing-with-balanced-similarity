function myLogInfo(str, varargin)  %('Testing [%s] on %d images ...', opts.modelType, length(ids))
% get caller function ID and display log msg
if nargin > 1
    cmd = 'sprintf(str';
    for i = 1:length(varargin)
        cmd = sprintf('%s, varargin{%d}', cmd, i);
    end
    str = eval([cmd, ');']);  %eval�Ĺ���Ϊ���ַ�������������ִ��
end 
[st, i] = dbstack(); %�������ö�ջ
caller = st(2).name;
fprintf('@%s: %s\n', caller, str);
end
