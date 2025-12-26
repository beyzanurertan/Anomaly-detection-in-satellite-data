
startTime = datetime(2025, 12, 18, 12, 0, 0);
stopTime = startTime + hours(1);
sampleTime = 0.1;

% Senaryoyu oluşturma
sc = satelliteScenario(startTime, stopTime, sampleTime);

% Uyduyu yörüngeye yerleştirme
% LEO Uydusu için gerekli 6 Kepler Parametresi:
semiMajorAxis = 6371000 + 500000; %  Yarı eksen (Dünya yarıçapı + 500 km)
eccentricity = 0;                 %  Basıklık (Dairesel)
inclination = 45;                 %  Eğim
RAAN = 0;                         %  Çıkış Düğümü Boylamı (0 derece)
argOfPeriapsis = 0;               %  En beri noktası argümanı (0 derece)
trueAnomaly = 0;                  %  Başlangıç konumu (0 derece)

sat = satellite(sc, semiMajorAxis, eccentricity, inclination, ...
                RAAN, argOfPeriapsis, trueAnomaly, ...
                'Name', 'H-Sat');

% simulasyon ve veri çekme
[position, velocity] = states(sat);

% Nominal yörünge hızında Jiroskop verisi hesaplama
orbitalPeriod = 2*pi*sqrt(semiMajorAxis^3 / 3.986e14);
nominalSpeed = 2*pi / orbitalPeriod; % Rad/sn

numSamples = length(position);
timeVector = (0:sampleTime:(numSamples-1)*sampleTime)';

% Nominal Jiroskop Verisi 
gyro_x = zeros(numSamples, 1);
gyro_y = -nominalSpeed * ones(numSamples, 1); % Pitch ekseni
gyro_z = zeros(numSamples, 1);

% veri kaydetme
Nominal_Veri = table(timeVector, gyro_x, gyro_y, gyro_z);
Nominal_Veri.Properties.VariableNames = {'Zaman', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'};

writetable(Nominal_Veri, 'Nominal_Yorunge_Verisi.csv');

disp('Nominal_Yorunge_Verisi.csv oluşturuldu.');
disp(['Uydunun yörünge periyodu: ', num2str(orbitalPeriod/60), ' dakika.']);

% Görselleştirme
v = satelliteScenarioViewer(sc);
play(v); 

% Grafik 1: Yörünge Konum ve Hız Değişimi (ECI Frame)
figure('Name', 'Orbit Dynamics', 'Color', 'w');
subplot(2,1,1);
plot(timeVector/60, position', 'LineWidth', 1.5); % Zamanı dakikaya çevirme
title('Uydu Konumu (ECI Koordinatları)');
ylabel('Konum (m)'); xlabel('Zaman (dk)');
legend('X', 'Y', 'Z'); grid on;

subplot(2,1,2);
plot(timeVector/60, velocity', 'LineWidth', 1.5);
title('Uydu Yörünge Hızı');
ylabel('Hız (m/s)'); xlabel('Zaman (dk)');
legend('Vx', 'Vy', 'Vz'); grid on;

% Grafik 2: Jiroskop / Açısal Hız Verisi (Sensör Çıktısı)
figure('Name', 'Sensor Data', 'Color', 'w');
plot(timeVector/60, Nominal_Veri.Gyro_X, 'r', 'LineWidth', 1.5); hold on;
plot(timeVector/60, Nominal_Veri.Gyro_Y, 'b', 'LineWidth', 1.5);
plot(timeVector/60, Nominal_Veri.Gyro_Z, 'k', 'LineWidth', 1.5);
title('Nominal Durum Jiroskop Verisi (Angular Velocity)');
ylabel('Açısal Hız (rad/s)'); xlabel('Zaman (dk)');
legend('Gyro X', 'Gyro Y (Pitch)', 'Gyro Z');
grid on;
ylim([-0.002 0.002]); % Y eksenini veriyi net görecek şekilde daralttık