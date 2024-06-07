SELECT 
    --Ids
        res.ID_Reserva AS 
            'Reserva',

    --Categorial Vars
        tipo_hab.Tipo_Habitacion_nombre AS 
            'Tipo_Habitacion',
        -- tipo_hab.Cupo AS 
        --     'CUPO',
        tipo_hab.Clasificacion AS 
            'Clasificacion_tipo_habitacion',
        paq.Paquete_nombre AS 
            'Paquete',
        can.Canal_nombre AS 
            'Canal',
        age.Agencia_nombre AS 
            'Agencia',
        -- hot.Empresa_nombre AS 
        --     'Hotel',
        -- hot.Franquicia AS 
        --     'Franquicia',
        sta.estatus_reservaciones AS
            'Estatus_res',

    -- --Montos
        hot.Habitaciones_tot AS 
            'Capacidad_hotel',
        res.h_num_per AS 
            'Numero_personas',
        res.h_num_adu AS 
            'Numero_adultos',
        res.h_num_men AS 
            'Numero_men',
        res.h_num_noc AS 
            'Numero_noches',
        res.h_tot_hab AS
            'Numero_habitaciones',
        res.h_tfa_total AS 
            'IngresoMto',

    --Fechas
        res.h_res_fec_ok AS 
            'FechaRegistro',
        res.h_fec_lld_ok AS 
            'FechaLlegada',
        res.h_fec_sda_ok AS
            'FechaSalida'
FROM iar_Reservaciones AS res

    -- --Tipo habitacion
    LEFT JOIN iar_Tipos_Habitaciones AS tipo_hab
        ON res.ID_Tipo_Habitacion = tipo_hab.ID_Tipo_Habitacion

    -- --Paquetes
    LEFT JOIN iar_paquetes AS paq
        ON res.ID_Paquete = paq.ID_paquete

    -- --Canal de venta
    LEFT JOIN iar_canales AS can 
        ON res.ID_canal = can.ID_canal

    -- --Agencias de viaje
    LEFT JOIN iar_Agencias AS age 
        ON res.ID_Agencia = age.ID_Agencia
    
    -- --Hotel
    LEFT JOIN iar_empresas AS hot
        ON res.ID_empresa = hot.ID_empresa

    -- --Estatus Reservaciones
    LEFT JOIN iar_estatus_reservaciones AS sta 
        ON res.ID_estatus_reservaciones = sta.ID_estatus_reservaciones
WHERE 
    --Ingreso mayor a 100
    res.h_tfa_total > 100

    --Noches mayores a 0
    AND res.h_num_noc > 0

    --Quitando las canceladas
    AND sta.estatus_reservaciones NOT IN ('NO SHOW', 'RESERVACION CANCELADA')

    --
    AND tipo_hab.Tipo_Habitacion_nombre NOT IN ('|')
ORDER BY res.h_tfa_total