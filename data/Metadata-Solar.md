Estas son las columnas que contiene SAJOMA-Solar-meta.csv y lo que representa cada una; he “traducido” el código del nombre a su significado práctico en un parque solar:

Columna	Qué mide	Sensor / altura	Orientación	Estadístico	Unidad

Timestamp	Fecha y hora de la lectura. Suele ser el sello de tiempo de un registro de 10 min.	—	—	—	—

Ch1_Anem_2.90m_W_Avg_m/s	Velocidad del viento (anemómetro).	2,90 m sobre el terreno	“W” (= canal físico West en la caja del logger)	Promedio	m s⁻¹

Ch13_Vane_2.90m_N_Avg_Deg	Dirección del viento (veleta).	2,90 m	“N” (= canal físico North)	Promedio	grados (0 = Norte, 90 = Este)

Ch16_Analog_2.00m_N_Avg_mb	Presión barométrica.	2,00 m	“N”	Promedio	mbar (hPa)

Ch17_Analog_2.00m_N_Avg_C	Temperatura del aire.	2,00 m	“N”	Promedio	°C

Ch18_Analog_2.00m_N_Avg_%RH	Humedad relativa.	2,00 m	“N”	Promedio	%RH

Ch20_Analog_2.80m_N_Avg_W/sqm	Irradiancia solar global (probable GHI; sensor horizontal).	2,80 m	“N”	Promedio	W m⁻²

Ch21_Analog_2.80m_N_Avg_W/sqm	Irradiancia de respaldo o segundo canal (p. ej. difusa o redundancia del GHI).	2,80 m	“N”	Promedio	W m⁻²

Ch102_Calculated___Avg_mm	Precipitación acumulada convertida a milímetros (cálculo interno a partir de un pluviómetro de cazoleta).	—	—	Promedio (aquí puede ser la suma del intervalo)	mm

Observaciones rápidas

La estructura “Ch #_Tipo_Altura_Orientación_Estadística_Unidad” es típica de dataloggers Campbell Scientific:
Tipo indica el sensor (Anem = anemómetro, Vane = veleta, Analog = entrada analógica genérica).
Altura se da en metros sobre el suelo donde está montado el sensor.
Orientación (N, W) suele ser solo el rótulo físico de la regleta del logger, no la orientación cardinal del instrumento.
Avg implica que en cada timestamp se guarda el promedio del periodo (en tu archivo se ve un paso de 10 min).
Unidad está al final.

Las irradiancias muestran valores negativos pequeños, típicos de lecturas nocturnas cuando el sensor/piranómetro se enfría y la electrónica del datalogger aplica un offset; puedes filtrarlos a cero para análisis de producción.

Si necesitas más detalle (factor de calibración, rango, etc.) habría que consultar la hoja de configuración (.CRBasic) o la ficha técnica del sitio.









