%inicializacao das variaveis
load numeros.dat
[K,N]=size(numeros);
map = [ 1 1 1;0 0 0];
P = numeros';
W = P*inv(P'*P)*P';

%contadores de SSIMVAL de cada rede neural
ssimSincSinal = 0;
ssimAsinSinal = 0;
ssimSincTanh = 0;
ssimAsinTanh = 0;

%aplica ruido na rede
r=0.450;
Pr = P;
for i=1:K
    for j=1:N
        m = rand(1,1);
        if m<r
           Pr(j,i)= -Pr(j,i);
        end
    end
end

%compara os valores da matriz base com a Hopfield sincrona com função sinal
for i=1:K
    Yant = Pr(:,i);
    Ynew = sign(W*Yant);
    cont =1;    
    while norm(Yant-Ynew)>0
        cont = cont+1;
        Yant = Ynew;
        Ynew = sign(W*Yant);
    end
    %reshape com os valores de Hopfield
    x=reshape(Ynew,5,7)';
    %reshape com os valores originais da matriz
    y=reshape(numeros(i,:),5,7)';
    %calcula o SSIM para as duas imagens e armazena no contador
    ssimSincSinal = ssimSincSinal + ssim(y,x);    
end
fprintf('O valor SSIM para a síncrona com função sinal é %0.4f\n',ssimSincSinal);

%compara os valores da matriz base com a Hopfield sincrona com função
%tangente hiperbolica
for i=1:K
    Yant = Pr(:,i);
    Ynew = tanh(W*Yant);
    cont =1;    
    while norm(Yant-Ynew)>0
        cont = cont+1;
        Yant = Ynew;
        Ynew = sign(W*Yant);
    end
    %reshape com os valores de Hopfield
    x=reshape(Ynew,5,7)';
    %reshape com os valores originais da matriz
    y=reshape(numeros(i,:),5,7)';
    %calcula o SSIM para as duas imagens e armazena no contador
    ssimSincTanh = ssimSincTanh + ssim(y,x);
end
fprintf('O valor SSIM para a síncrona com função tanh é %0.4f\n',ssimSincTanh);

%compara os valores da matriz base com a Hopfield assincrona com função
%sinal
for i=1:K
    Yant = Pr(:,i);
    pos = ceil(rand(1,1)*N);
    Ynew = Yant;
    Yaux = sign(W*Yant);
    Ynew(pos)=Yaux(pos);
    cont =1;
    while norm(Yant-Yaux)>0
        cont = cont+1;
        pos = ceil(rand(1,1)*N);
        Yant = Ynew;
        Yaux = sign(W*Yant);
        Ynew(pos)=Yaux(pos);
    end
    %reshape com os valores de Hopfield
    x=reshape(Ynew,5,7)';
    %reshape com os valores originais da matriz
    y=reshape(numeros(i,:),5,7)';
    %calcula o SSIM para as duas imagens e armazena no contador
    ssimAsinSinal = ssimAsinSinal + ssim(y,x);
end
fprintf('O valor SSIM para a assíncrona com função sinal é %0.4f\n',ssimAsinSinal);

%compara os valores da matriz base com a Hopfield assincrona com função
%tangente hiperbolica
for i=1:K
    Yant = Pr(:,i);
    pos = ceil(rand(1,1)*N);
    Ynew = Yant;
    Yaux = tanh(W*Yant);
    Ynew(pos)=Yaux(pos);
    cont =1;
    while norm(Yant-Yaux)>0
        cont = cont+1;
        pos = ceil(rand(1,1)*N);
        Yant = Ynew;
        Yaux = sign(W*Yant);
        Ynew(pos)=Yaux(pos);
    end
    %reshape com os valores de Hopfield
    x=reshape(Ynew,5,7)';
    %reshape com os valores originais da matriz
    y=reshape(numeros(i,:),5,7)';
    %calcula o SSIM para as duas imagens e armazena no contador
    ssimAsinTanh = ssimAsinTanh + ssim(y,x);
end
fprintf('O valor SSIM para a assíncrona com função tanh é %0.4f\n',ssimAsinTanh);
