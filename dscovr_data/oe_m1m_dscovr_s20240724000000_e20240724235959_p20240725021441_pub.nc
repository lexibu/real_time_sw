CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240724000000_e20240724235959_p20240725021441_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-25T02:14:41.268Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-24T00:00:00.000Z   time_coverage_end         2024-07-24T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy �   "          Arff@@  �`��@���A�
=C��@@  �[�
@��
A�=qC�&f                                    By ��  �          Ax  @qG��h  @|(�Ak�C�G�@qG��c�@��A���C�j=                                    By �L  
�          A�(��.{�k�
@��A��HC��\�.{�d(�A33A���C���                                    By ��  
�          A����L(��^ffA'�
B\)C����L(��S�A5�B{C�7
                                    By Ԙ  
�          A����\)�EA(  B�Cy���\)�;
=A3�
B&�
Cw��                                    By �>  �          A�
=��=q�J=qA(z�B�Cy#���=q�?�A4��B$Cx�                                    By ��  
�          A�  ��(��G\)A6=qB��Cq{��(��;�AB=qB'
=Co�
                                    By! �  
�          A�(���  �?
=A333B��Cm����  �3�A>�RB%�
ClB�                                    By!0  
�          A�����
=�L(�A5�B
=Cr:���
=�@Q�ABffB%�Cp��                                    By!�  
�          A�G���ff�K
=A-p�B(�Cs���ff�?�
A9�B!��Cq�R                                    By!,|  T          A�z�����P��A&�\BG�Ct������EA3\)BQ�Cs�                                     By!;"  �          A�ff����R=qA'
=B�RCu�=����G33A4  B�Ct�)                                    By!I�  �          A�(���p��Y��A0��B
=Cu�{��p��MA>ffBG�Ct^�                                    By!Xn  �          A����\)�f�HA333B��Cw� ��\)�[
=AA��B�Cvff                                    By!g  �          A�����\)�R�\A@��B�
Cr���\)�EAM�B(Cp�\                                    By!u�  �          A�����
=�W33A@  B��Cu.��
=�JffAMp�B)Q�Cs��                                    By!�`  
�          A�=q�ȣ��Yp�A;�B\)Cu:��ȣ��L��AIG�B%Cs�f                                    By!�  �          A�z������Z=qA=�B��Cv{�����Mp�AK�B'�\Ct�                                    By!��  �          A�����
�\(�A8Q�BffCv�f���
�O�AFffB$�Cu��                                    By!�R  
�          A������^�\A6=qBG�CwT{����R{ADz�B"�Cv#�                                    By!��  �          A����
=�\  A4��B�RCwk���
=�O�AB�HB"��Cv8R                                    By!͞  
(          A�����
�]�A5G�B�HCw޸���
�P��AC�B"�HCv��                                    By!�D  
�          A�z��ʏ\�]��A0��B��Cuk��ʏ\�QG�A?
=B��Ct+�                                    By!��  "          A�Q������]p�A/�
B{Cu.�����Q�A>{B�Cs�                                    By!��  �          A�����G��]p�A.=qBz�Cu�\��G��QG�A<��BffCtQ�                                    By"6  T          A�G��ҏ\�Y�A/�B33Ct)�ҏ\�L��A=�B
=Cr�                                    By"�  
�          A�33���W33A0��Bz�Cs�
���J�RA?
=BG�Cr5�                                    By"%�  
�          A�p�����X��A+�BffCt������L��A:{BffCsc�                                    By"4(  
�          A�Q�����b�\A#�B{Cy
����V�HA2�RB��Cx�                                    By"B�  T          A�Q����R�^=qA ��BCx�=���R�R�RA/�BQ�Cwz�                                    By"Qt  �          A�  ����[33A
=B\)Cw����P  A)B�
Cu�                                    By"`  T          A�G������X(�A��B��Cv�����L��A+33B(�Cu��                                    By"n�  
�          A�
=�����[33A��B��CvE�����O�
A+�BQ�Cu�                                    By"}f  
�          A��H�ҏ\�Z�HA�B�CtL��ҏ\�O\)A,��Bz�Cs�                                    By"�  T          A�ff�љ��Yp�ABCtE�љ��MA,��B(�Cs                                    By"��  �          A�  ���
�X��A (�B(�Ct�=���
�L��A/
=B�RCs��                                    By"�X  �          A����Ӆ�]A Q�B�Ct� �Ӆ�Q�A/�B�Cs@                                     By"��  �          A�Q���
=�Tz�A&�RB�CrL���
=�H(�A5G�Bz�Cp�)                                    By"Ƥ  T          A������N�RA'\)BQ�Cp+�����B=qA5��BffCn�)                                    By"�J  T          A�����{�X��A=qB�Cq5���{�L��A-p�B�
Co��                                    By"��  
�          A�����Tz�A  A��Cm�3���H��A*�HBp�Clu�                                    By"�  "          A����z��O�A$��BG�Cl���z��C
=A3�B
=Ck                                      By#<  �          A�G����RffA"�RB  Cn�f���F{A1��B(�CmW
                                    By#�  "          A�z���p��RffA%�B	\)Cp����p��EA4  B�HCo                                    By#�  
�          A�=q���H�W33A\)Bz�Cs�����H�J�HA.�RB�\Cr�\                                    By#-.  "          A�����{�P  A
=B	��Cs����{�C�
A-�B�Cr@                                     By#;�  
�          A������H�P  A!��B�RCt���H�C�A0��B�Cr�
                                    By#Jz  
�          A�33��p��N�HA z�B��CuaH��p��B=qA/�BG�Cs�q                                    By#Y   T          A�ff���R�N{A"ffB�
Cw����R�Ap�A1p�B�\Cu��                                    By#g�  
Z          A�����R�H��A*=qB
=Cxh����R�;\)A8��B)  Cw                                    By#vl  
�          A�G���=q�S�
A�\B	{Cy\��=q�G�A*{B=qCw��                                    By#�  T          A�G����H�[
=A��B��Cy�H���H�N�HA(��B�Cx��                                    By#��  t          A���<���j�HA=qA��C�S3�<���_\)A�B�C�
=                                    By#�^  
(          A�����
=�M��A  A��RCrc���
=�A�A\)B�HCq                                      By#�  �          A�=q�����]G�A
=A�Q�C{c������QA�B�HCzu�                                    By#��  "          A��H�����@Q�A"{B�\Cv33�����333A0z�B&�RCt��                                    By#�P  "          A�����33�;
=A�BG�Cuk���33�.{A-B'ffCs��                                    By#��  "          A�{�p  �Y��A33A�\)C~���p  �N�RA�
B	��C}��                                    By#�  T          A��?���x��@��A��C�?���p��@�A�\)C��                                    By#�B  B          A��\@33�m�@߮A��
C�j=@33�c�A=qA�C���                                    By$�            A�p�?Tz��g33@�{A��
C���?Tz��\��A��B�C��
                                    By$�  "          A��
�N�R�f�\@�G�A�ffC���N�R�\  A�\B
=C�b�                                    By$&4  �          A�������\  A ��A�ffCq�����P��A=qA���Cpn                                    By$4�  �          A���>#�
�}��@��HA̸RC�H�>#�
�s
=Ap�A�C�L�                                    By$C�  t          A�z`\)�z=q@��HA�Q�C�� ��\)�o33AG�A�G�C�c�                                    By$R&  �          A�{�   �33@���A���C�n�   �t��A��A��
C�G�                                    By$`�  �          A��Ϳk��|��@��A���C�S3�k��s
=A�HA�{C�B�                                    By$or  �          A��Ϳ�33�x��@��
A��C�����33�o
=A{A��C�ٚ                                    By$~  B          A��R�p��x��@�  A�  C����p��o
=A (�A߅C���                                    By$��  �          A��H��\)�w
=@��A�  C����\)�l��A�RA��C��H                                    By$�d  T          A�z��Q��u�@���A�ffC���Q��j�RAffA�Q�C��)                                    By$�
  �          A��R���v�H@޸RA�Q�C��3���l��A�A�Q�C���                                    By$��  T          A�
=���t��@��A�(�C������j{A	�A�ffC�z�                                    By$�V  "          A��
���
�|z�@�  A�Q�C�<)���
�r�\A ��A�
=C��                                    By$��  T          A�G��c�
�}@��
A�33C�e�c�
�s
=A
=A�(�C�S3                                    By$�  �          A��ÿ�Q��~�\@�ffA�p�C��f��Q��tz�A z�A�Q�C�aH                                    By$�H  �          A�ff�Z�H�|(�@��RA��C�� �Z�H�s�@�G�A�(�C���                                    By%�  �          A�Q����s�@�A��HC��
����hQ�Ap�A�z�C��=                                    By%�  
(          A�{���
�s
=@��HA�Q�C�g����
�g�A{A�(�C�`                                     By%:            A��׿����x  @ᙚA�
=C�˅�����mG�A�A���C���                                    By%-�  �          A�=q����w�@޸RA�G�C�� ����l��A��A�\)C��                                    By%<�  "          A�33���{�
@��
A��\C��H���q@�\)A֏\C��
                                    By%K,  T          A���9���}p�@��
A�ffC�Ф�9���s�
@�  A�Q�C��q                                    By%Y�  �          A����G��s�A   A��C�˅��G��g33AG�B\)C���                                    By%hx  
�          A�  �
=q�n�\@�33A�  C��ÿ
=q�b�RAffB �C��                                    By%w  T          A�������l(�@���A�Q�C�j=�����`  A�BC�b�                                    By%��  "          A���
=�l��@�33A�\)C���
=�`��AffBffC�˅                                    By%�j  �          A��׿L���g�
@�  Aڏ\C�j=�L���\  Az�B{C�T{                                    By%�  t          Az�H����\��@�p�A�C�� ����Q�A
=qB��C��                                    By%��  �          At(��.{�I�A	B	33C���.{�<z�A�B=qC��{                                    By%�\  �          As
=����MG�@���A��C�����@��A��B��C��{                                    By%�  �          Ap(��]p��\��@~{Ayp�C�3�]p��V=q@��A��C�                                    By%ݨ  T          As33�!G��c�@���A�\)C��R�!G��\  @��HA�\)C���                                    By%�N  B          As��E��X��@�z�A��RC����E��N�R@�(�A�\C�L�                                    By%��  4          At������S�
@�\A��C�� ����G�
A��B�C��3                                    By&	�  
�          Au�����e�@�p�A�{C�@ ����]�@�  A���C��                                    By&@  
�          Av=q�%��hQ�@��A�ffC����%��`��@�p�A�33C�˅                                    By&&�  "          Au����j�H@���As
=C�Y�����c�@���A���C�<)                                    By&5�  �          Av�\��33�g33@���A���C�����33�^ff@���A�ffC��q                                    By&D2  
�          Ax�����j=q@��HA�
=C��3���a��@�\)A���C�˅                                    By&R�  "          A}���
�o33@��RA��
C������
�fff@�(�A�  C�xR                                    By&a~  �          A{��(��o\)@�{Aw\)C�^��(��g�@�(�A�C�8R                                    By&p$  �          A|����H�qG�@w�Ac33C�p���H�j{@�=qA��C�L�                                    By&~�  �          A}p�>�p��mG�@�G�A�C��{>�p��c\)@�
=A��HC��)                                    By&�p  "          A{�>����t��@aG�AO
=C���>����n=q@�Q�A���C���                                    By&�  �          A|(�?���t��@i��AV{C��q?���m��@���A�z�C��                                    By&��  "          Ay�?��\�u��@)��Az�C���?��\�p(�@�p�Aw�
C��3                                    By&�b  �          A~�R?
=q�y��@C33A0��C��q?
=q�s�@�33A�ffC��                                    By&�  �          A�
>��x��@c�
AN=qC�:�>��q�@��A�\)C�=q                                    By&֮  
�          A�{    ��@�33Af{C�H    �w�@�{A��C�H                                    By&�T  T          A�?�G��y@�33A�
=C�P�?�G��l��A
ffA�C�p�                                    By&��  T          A���?����up�@�(�A�  C���?����h(�A�\A��\C��3                                    By'�  
�          A��H@;��|��@�p�A�ffC�@ @;��o�A  A���C���                                    By'F  �          A��
@)���|��@��HAʸRC��H@)���n�HA�HA�p�C�f                                    By'�  
�          A��R?��R�{
=A  A�Q�C���?��R�l  AG�BC���                                    By'.�  �          A�{@<(��w\)A\)A�Q�C�aH@<(��hQ�Az�B�C��
                                    By'=8  
Z          A���@(Q��}G�@�  A�z�C��R@(Q��o�A�A��C��)                                    By'K�  "          A���@ff�{�@�=qA�  C��@ff�l��A
=B ��C��                                    By'Z�  �          A��@   �yA Q�A��C���@   �j�HA�B�C��{                                    By'i*  
�          A�\)@8���yp�@���A��HC�@ @8���j�RA{B p�C���                                    By'w�  T          A�(�@@  �zff@���A�ffC�k�@@  �l��A�\A�\C���                                    By'�v  
�          A�p�@33�o�A (�A���C�]q@33�`��A�BC���                                    By'�  
          A�\)?�=q�^�\A��A�=qC�7
?�=q�NffA (�B�
C�b�                                    By'��  
�          Apzᾮ{�=�A�
B  C�/\��{�,  A'�B0�HC��                                    By'�h  T          ApQ�?���AG�A  B�
C��H?���0��A (�B'��C���                                    By'�  
�          Ajff?Tz��6ffA=qB��C�
?Tz��%G�A%G�B3�C�N                                    By'ϴ  
Z          Ajff>����6�HA�HB��C��3>����%��A&{B433C���                                    By'�Z  �          Ag\)?Tz��:�HA�
B�RC�f?Tz��*�RA�B)
=C�7
                                    By'�   
�          AeG�?0���2=qA�RBp�C�Ǯ?0���!p�A!��B3�HC��R                                    By'��  
�          Aq?����=p�AQ�B��C��
?����+�A(z�B1Q�C���                                    By(
L  
�          Ax��>��IG�AQ�Bz�C��>��7�
A&{B(33C�4{                                    By(�  T          A}��?z�H�L(�A��B
=C�1�?z�H�:{A*�HB)�
C�g�                                    By('�  	�          Az�\?:�H�H��A��Bp�C���?:�H�6�RA+\)B,ffC��3                                    By(6>  
(          A}G�>�z��N{A33B{C���>�z��<  A)��B(33C��{                                    By(D�  �          A�������TQ�A��B��C��=�����A�A,  B&=qC��                                    By(S�  T          A~�H�#�
�P  A\)B=qC����#�
�=A*ffB'��C��f                                    By(b0  
H          A��\?z�H�I�AB{C�:�?z�H�5��A4  B2�C�w
                                    By(p�  4          A�  ?(��Pz�A ��Bz�C�W
?(��<z�A8  B1{C�z�                                    By(|  	�          A�Q�>�ff�Q�A'\)B33C��)>�ff�<��A>�RB4��C�
                                    By(�"  
�          A��R>��
�M��A'�
B�HC��R>��
�8z�A>�RB7��C���                                    By(��            A�G�>\�QA$(�B(�C��
>\�<��A;�B333C��\                                    By(�n  
(          A���>���T  A%�B(�C���>���>�HA=B3Q�C��q                                    By(�  �          A��H?
=q�Z�HA�B�C�"�?
=q�F�\A6�RB*ffC�@                                     By(Ⱥ  
(          A���?��H�^�\A\)B�C�|)?��H�J�HA0��B#�
C���                                    By(�`  
�          A�\)@/\)�n�R@�{A���C�1�@/\)�]A�RB	�\C���                                    By(�  �          A�33@\(��u��@�  A��C�O\@\(��f�RA��A��
C��{                                    By(��  
�          A�z�@`  �t��@��
A���C�o\@`  �f=qA�HA��C��3                                    By)R  t          A��\@Dz��hz�@��A��
C���@Dz��Xz�Az�B=qC�b�                                    By)�  f          A�Q�@1G��g33Ap�A�ffC�j=@1G��U��A��BC�ٚ                                    By) �  	�          A�
=?=p��N�\A z�B=qC��f?=p��9G�A8��B3ffC���                                    By)/D  "          A�33?�=q�V�RAz�B  C�N?�=q�AA5B,=qC��                                    By)=�  �          A��?�ff�[�A33B	z�C�8R?�ff�G\)A1G�B%C���                                    By)L�  	�          A�
=@��f�HA{A��C�|)@��Tz�A!B�\C��q                                    By)[6  "          A�p�@z�H�v=q@��A��C�%@z�H�g
=A
=A�G�C��
                                    By)i�  �          A��R@��R�|��@��
A�ffC�N@��R�o�
@��AǮC��R                                    By)x�  
�          A�\)@�{�|��@���A��\C��@�{�o�
@�A��
C��                                    By)�(  	�          A�Q�@�p��|��@���A�p�C��f@�p��pQ�@�  A���C��                                    By)��  
�          A��\@����~�R@�G�Ae��C�&f@����t  @�G�A�Q�C���                                    By)�t  "          A�
=@���~�H@��\At(�C��@���s�@ҏ\A��C��H                                    By)�  
�          A��
@��
�x��@�(�A�z�C��3@��
�k
=@�\A���C�+�                                    By)��  
�          A��@��H�}��@��A��\C�}q@��H�p(�@�A�33C��                                    By)�f  �          A�  @�������@�p�A�Q�C�AH@����u@�\)A�p�C��                                    By)�  T          A�p�@�
=��(�@�33A�z�C�R@�
=�u@�p�A��C���                                    By)��  
�          A��
@������R@�=qA�\)C�\@����s
=@�(�A��C���                                    By)�X  "          A���@�p��|��@�
=A�=qC���@�p��o\)@�Q�A��
C�)                                    By*
�  
�          A��
@����xQ�@�{A�33C�,�@����j�\@�{A̸RC���                                    By*�  
�          A�@��R�vff@�\)A�=qC��H@��R�hz�@�
=A͙�C�H�                                    By*(J  
l          A��@����t  @��A�G�C���@����e�@�G�A���C��                                     By*6�  
(          A�G�@�z��uG�@���A�  C���@�z��g33@��A��C�:�                                    By*E�  	�          A�  @�G��z{@�
=A��HC��@�G��k�@���A�(�C���                                    By*T<  �          A�{@�\)�z�\@�A��C���@�\)�l  @��A�\)C�q�                                    By*b�  
�          A�(�@�Q��y��@���A���C��H@�Q��j=qAp�A߮C��                                    By*q�  
�          A�p�@tz��v=q@��HA���C���@tz��ep�A
{A�p�C�q�                                    By*�.  
(          A���@���|��@�
=A��RC��@���l  A	�A�p�C�                                    By*��  �          A���@��|Q�@��A��
C�E@��l  A�RA�ffC��                                    By*�z  T          A��R@�=q�|  @�p�A�Q�C���@�=q�l  A��A���C��                                    By*�   
�          A�G�@����r=q@ə�A�ffC�/\@����a�Ap�A��HC���                                    By*��  
�          A�(�@�(��t��@�\)A�(�C���@�(��d  A��A�z�C�U�                                    By*�l  B          A�@�z��{�@��A���C�t{@�z��k\)A��A�p�C�\                                    By*�            A���@�(��~{@��A�\)C�\)@�(��n�R@��HA�z�C���                                    By*�  �          A�z�@�p��
=@�  A�z�C�c�@�p��q�@�RA��
C���                                    By*�^  "          A�  @��
�x��@��
A�(�C��f@��
�j�H@���A�C�                                    By+  �          A�\)@��H�s�@�=qA�p�C���@��H�e��@�{A�33C�33                                    By+�  "          A�Q�@�G��o�@�Q�A�
=C��)@�G��`z�@�33A���C�U�                                    By+!P  "          A��@���i@��
A�  C���@���[�@�p�A��C��3                                    By+/�  
�          A���@�ff�l��@�
=A�\)C�  @�ff�_33@ٙ�AÅC��                                    By+>�  
�          A�(�@�{�lz�@��RA��
C�)@�{�^ff@ᙚA�Q�C���                                    By+MB  �          A��\@�(��l��@�p�A��
C�n@�(��\��@���A�=qC��                                    By+[�  �          A��@��
�o�
@�Q�A��HC���@��
�`��@���A�\)C�T{                                    By+j�  "          A���@�p��g�@���A�ffC�)@�p��YG�@��HA̸RC���                                    By+y4  
�          A���@�\)�iG�@�  A{�
C��q@�\)�\(�@�33A��C���                                    By+��  "          A�{@�
=�k33@���A^�\C��\@�
=�_
=@�p�A���C�G�                                    By+��  
(          A���@�{�c�
@��\Af�HC�  @�{�W�@���A��RC���                                    By+�&  
�          A�@��bff@�=qAw
=C���@��Up�@�z�A�p�C�L�                                    By+��  
�          A|��@��H�[\)@�Q�Az=qC��3@��H�N�R@ȣ�A�
=C���                                    By+�r            A|��@���T��@���A���C�O\@���Ep�@�A�z�C�.                                    By+�  e          A�=q@�R�_�@=p�A*�\C��f@�R�U@�G�A���C�,�                                    By+߾  
�          A���@�G��h��@A�C���@�G��`z�@���A���C�S3                                    By+�d  
�          A�\)@����pz�@�{A��C��
@����ap�@�Aȣ�C��q                                    By+�
  "          A��R@���s�@�{A�ffC��q@���d��@�
=A��
C�`                                     By,�  "          A��@���l��@�z�A�\)C���@���]�@�A���C�z�                                    By,V  
�          A�
=@����o
=@��HA�=qC��R@����_�@��HA�  C��f                                    By,(�  
{          A��@���lQ�@�p�A���C���@���[\)@���A�
=C���                                    By,7�  3          A�ff@�p��lQ�@��HA�
=C�W
@�p��\��@�\A���C�                                    By,FH  �          A�ff@�z��h��@�
=A�C�t{@�z��YG�@�A��
C�+�                                    By,T�  
�          A�Q�@�ff�m��@��A��C���@�ff�]@�(�AЏ\C���                                    By,c�  
9          A�p�@�(��j�R@���A�  C���@�(��[�@��A�p�C�Q�                                    By,r:  T          A���@��h��@�(�Ab�HC��f@��[\)@�(�A�(�C�9�                                    By,��            A��@��
�g
=@��Ac\)C�J=@��
�Yp�@��A��C��                                    By,��            A�G�A   �f�H@k�AIG�C���A   �Zff@�{A���C�0�                                    By,�,  �          A���A\)�c�@H��A,(�C�` A\)�Xz�@�(�A�33C��                                    By,��  u          A�p�A�R�b{@/\)A��C�"�A�R�X  @�\)A��HC���                                    By,�x  3          A�(�A�`Q�@J�HA+�
C���A�U�@���A�  C�0�                                    By,�  
�          A���Ap��b�H@<(�A=qC�T{Ap��X(�@�ffA�C��
                                    By,��  T          A���A�\�c�
@A�A"ffC�` A�\�X��@�=qA�  C�f                                    By,�j  �          A�A
=�d��@^{A:{C��A
=�Xz�@�Q�A��\C��R                                    By,�  "          A�{A��hz�@w�AO�C��A��[
=@ƸRA��RC�޸                                    By-�  
q          A��
A���^{@Q�@�
=C�� A���T��@�z�Az�HC�5�                                    By-\            A�p�A#33�Y�@��A z�C�k�A#33�Pz�@�(�Ay�C�f                                    By-"  
�          A�p�A"�H�Yp�@!G�A
=C�l�A"�H�O�@�  A�ffC�\                                    By-0�  �          A�  A!G��\  @'�A�C��A!G��Q�@�(�A�p�C���                                    By-?N  
Z          A��A ���[�@3�
A=qC��A ���P��@��\A���C��                                    By-M�  �          A�(�A(��^�H@8Q�Ap�C���A(��T  @�A��
C�1�                                    By-\�  T          A�=qAz��fff@mp�AFffC���Az��Y�@��HA�G�C�w
                                    By-k@  �          A�  A
=�jff@~�RAUG�C��RA
=�\(�@��A�z�C�^�                                    By-y�  �          A��A��g�
@~{AV�RC��=A��Y��@�z�A��C���                                    By-��  "          A�z�@�
=�hQ�@���Av{C���@�
=�X��@�ffA�(�C��R                                    By-�2  
�          A��H@��
�j{@�Q�A�\)C���@��
�X(�@�ffA�  C��f                                    By-��  
�          A��R@�z��i�@��A��RC�@�z��X  @�Aՙ�C���                                    By-�~  �          A�=q@�ff�i��@�\)A�(�C�  @�ff�XQ�@�{A�33C�                                    By-�$  
x          A���@�G��k\)@�\)A��RC�,�@�G��Z�\@�
=A�  C��                                    By-��  
�          A��\@��\�p��@�ffA��\C�T{@��\�\  A�
A�ffC�&f                                    By-�p  �          A�33@�
=�q@ƸRA��C���@�
=�\z�A(�A��
C�^�                                    By-�  �          A��@��R�s
=@��A���C���@��R�^ffA��A�RC���                                    By-��  �          A��
@�\)�s�@�(�A��
C��@�\)�_33A�A�{C���                                    By.b  �          A���@��H�r�\@�A��C���@��H�]�AQ�A�\C��                                    By.  
�          A�Q�@�(��q�@\A�G�C�ff@�(��\  A
�RA�\C�<)                                    By.)�  �          A�z�@����r=q@�G�A�  C�/\@����]�A
ffA�C�                                      By.8T  
�          A��H@��
�s33@�
=A�Q�C���@��
�^�RA��A��
C��=                                    By.F�  "          A��H@�G��rff@�A�G�C��@�G��^{A�A���C��                                    By.U�  �          A���@�ff�r�\@�p�A�(�C�K�@�ff�^�HA�A�C�q                                    By.dF  T          A�ff@�{�qp�@���A�G�C�T{@�{�]p�A�RA��C�.                                    By.r�  
�          A�33@���pz�@��A���C��3@���]p�@�Q�Aڣ�C�b�                                    By.��  �          A��\@�\)�qp�@��RA�=qC�e@�\)�_�@�z�AЏ\C�%                                    By.�8  "          A��@�z��e@��
A��C�ff@�z��O�
A
ffA�\)C�N                                    By.��  �          A��
@�
=�Y@�G�A�G�C�� @�
=�?
=A"�\B�C��q                                    By.��  	�          A���@���UG�@��A�ffC���@���9�A%�B��C��
                                    By.�*  
�          A�{@�  �V{@���Aԏ\C�>�@�  �=G�A  B
=C�w
                                    By.��  	�          A|  @��H�X��@�
=A��\C�7
@��H�C�
A=qA�G�C�Ff                                    By.�v  	�          Aw�@�ff�U��@��
A�Q�C��\@�ff�A�@���A���C���                                    By.�  
�          Aw33@��\�V=q@���A��
C���@��\�C\)@�{A�\C���                                    By.��  T          Av=q@�Q��Tz�@�A���C�,�@�Q��A@��HA�{C�1�                                    By/h  T          Au��@�(��S33@��A��
C�s3@�(��@��@��A�=qC�z�                                    By/  
�          Aw33@�\)�R�R@���A�=qC�
@�\)�@z�@�{A�  C�&f                                    By/"�  �          Az{@�G��Y@���AnffC���@�G��Ip�@���A�C��R                                    By/1Z  
�          A�
@ڏ\�\z�@��
A}�C�,�@ڏ\�K
=@�p�A�G�C�"�                                    By/@   �          A��\@�\)�\��@�  Au�C�j=@�\)�K�
@�=qA�33C�^�                                    By/N�  �          A�Q�@�\)�_
=@�{Ar=qC��@�\)�M�@�G�A�z�C�˅                                    By/]L  
�          A
=@�Q��^�R@���A�=qC�q@�Q��Lz�@�(�A��HC�                                    By/k�  �          Ax��@����\��@��RA�\)C��3@����K
=@ᙚA��
C���                                    By/z�  T          A}��@ҏ\�\��@�(�AqC���@ҏ\�K�
@�\)A�G�C���                                    By/�>  
�          A�{@����b�\@�33A|Q�C���@����P��@���A�ffC�s3                                    By/��  "          A}@�p��b=q@��RAuC�]q@�p��P��@���AͮC�5�                                    By/��  T          Ayp�@��\�`  @���An�HC��)@��\�O
=@�{A�33C���                                    By/�0  �          A{\)@���c33@Z�HAI�C�H@���T  @���A�z�C��
                                    By/��  "          A{�@�z��a��@mp�AZffC�U�@�z��QG�@�A�
=C�q                                    By/�|  T          Ay�@�Q��]p�@\(�ALQ�C�.@�Q��N{@��
A��C��R                                    By/�"  T          Az{@�=q�Tz�@�ffAy�C��@�=q�B�H@أ�A�\)C��                                    By/��  
Z          Av�H@�(��YG�@\)Ao�C�#�@�(��H(�@��
A�\)C��                                    By/�n  T          At��@�33�V�\@q�Ad��C��=@�33�F{@���AŅC��3                                    By0  T          At��@ƸR�W�
@n�RAa��C�Z�@ƸR�G\)@��
Aģ�C�<)                                    By0�  
�          Av{@�Q��X  @_\)AQC�޸@�Q��H(�@�z�A�Q�C��)                                    By0*`  C          Au�@�=q�Y�@'
=A�C��@�=q�L(�@��A�Q�C��                                     By09  e          Au�@�ff�Y�@?\)A4(�C���@�ff�J�R@�{A�ffC�z�                                    By0G�  
�          AuG�@�{�Y@6ffA+\)C���@�{�K�
@�=qA�Q�C�g�                                    By0VR  "          Ar�R@����W
=@>�RA5G�C���@����H��@�p�A��C��H                                    By0d�  T          Ar�\@����U��@7
=A.=qC��@����G�@�G�A��
C�Ф                                    By0s�  
�          As�@�
=�W�@8��A/
=C�Ф@�
=�I��@��A���C��
                                    By0�D  3          Av{A��?\)@s33AeG�C�EA��.�R@�ffA�(�C��=                                    By0��  T          Au�AG��<z�@w
=Ah��C��\AG��+�@�
=A��C�                                      By0��  T          AyG�A��B�R@o\)A^�\C�33A��1�@�ffA���C�q�                                    By0�6  T          Ax��A=q�@z�@hQ�AX  C��qA=q�0(�@��A��C���                                    By0��  �          Aw
=A\)�A�@Tz�AF�\C�<)A\)�2�\@���A�{C�b�                                    By0˂  T          Av�\Aff�@��@fffAX  C�:�Aff�0Q�@���A�
=C�w
                                    By0�(  u          Au�Ap��A@P��ADz�C�3Ap��2ff@�  A��C�7
                                    By0��            AuAz��C�@C33A7�C���Az��4��@�=qA�=qC��=                                    By0�t  �          Au�A33�D��@@��A4��C�� A33�6=q@��A��C��\                                    By1  �          Av=qA
=�C\)@^�RAQG�C���A
=�3
=@�Q�A�C���                                    By1�  T          AuG�A��A@\��AP  C���A��1p�@��RA�
=C�R                                    By1#f  
�          Av=qA�
�8��@�A	p�C�3A�
�,��@��RA��C�
=                                    By12  "          Aw
=A.=q�,(�?��H@�G�C���A.=q�!G�@�\)A~ffC��R                                    By1@�  �          At(�A
=�6{@z�@��C���A
=�*�\@�\)A��RC�}q                                    By1OX  
(          Aqp�A ���G�
@'�A (�C�ffA ���:{@���A�\)C�XR                                    By1]�  
          Ao
=@�R�L��@-p�A'\)C��{@�R�>ff@�A�z�C��)                                    By1l�  3          AlQ�@��J{@P  AK\)C�^�@��9�@�A�33C�ff                                    By1{J  r          Ah��@�{�G\)@3�
A2ffC��@�{�8��@��RA��RC��H                                    By1��  �          Ah��@�G��G\)?���@�{C�,�@�G��;�@�(�A�ffC���                                    By1��  
�          Al��@�\)�K
=?��
@�C�B�@�\)�@��@���A��\C���                                    By1�<  
�          Aq@�{�L(�?��@���C���@�{�B=q@�A��\C��R                                    By1��  �          Av�H@���T��@8��A,��C�U�@���E�@�Q�A�p�C�E                                    By1Ĉ  
Z          Av�\@�ff�R�R@5A*=qC�� @�ff�C\)@�ffA�C���                                    By1�.  �          Av{@�33�P��@:�HA/33C��@�33�A�@�Q�A��C�
                                    By1��  C          Au��@�\)�Q@0  A%G�C���@�\)�B�R@��A��C�˅                                    By1�z            Aup�@�(��N{@*=qA (�C��)@�(��?33@�\)A��C��{                                    By1�   �          As33A���Fff@8Q�A.�RC�� A���7
=@��HA��HC���                                    By2�  
�          As
=A��E@4z�A+\)C�A��6ff@�G�A�
=C�#�                                    By2l  T          Atz�A	�D��@7
=A,��C���A	�5�@�=qA�
=C��H                                    By2+  
�          Au�A���A@G
=A;33C��A���1p�@���A�33C�H�                                    By29�  �          Ar�RA��B�H@K�AA��C�nA��2=q@��
A�{C��f                                    By2H^  
�          Ar�R@�\)�HQ�@I��A?�C�AH@�\)�7�@�A��
C�h�                                    By2W  "          Ao�@�p��H(�@333A,��C��H@�p��8z�@��HA��C��                                    By2e�  T          Aa@����7�@QG�AVffC�� @����&�R@�=qA���C��                                    By2tP  T          Aa�@�Q��7�@Z�HA_�
C���@�Q��&{@�
=A��C��                                    By2��  
�          Aa��@�(��8��@O\)AT��C�G�@�(��(  @��\A�p�C��\                                    By2��  �          A`��@�(��4z�@^{Ad��C��@�(��"�R@��A�  C�o\                                    By2�B  
Z          Ag\)@���Ep�@2�\A1�C���@���5��@��\A���C��q                                    By2��  
�          Ao�@��
�G�
@I��ABffC��\@��
�6�\@�\)A�Q�C��)                                    By2��  T          Ar=q@�\)�G
=@N{AD��C�W
@�\)�5p�@�G�A�z�C���                                    By2�4  
�          ApQ�@���E@J�HAC
=C�P�@���4Q�@�\)A��
C���                                    By2��  T          ApQ�@�p��E@FffA>�RC�S3@�p��4z�@�p�A��C���                                    By2�  �          Aq@��M�@0��A(z�C�%@��<��@��RA��C�0�                                    By2�&  �          At  @�p��Q�@#�
A�\C�˅@�p��A��@��\A�C��                                    By3�  "          Aq��@��H�O�@��AC���@��H�@��@���A�C��\                                    By3r  �          Ao
=@�Q��Q��?�=q@\C�{@�Q��E��@��A�G�C�˅                                    By3$  
�          Ar�\@�ff�P(�@33@���C��f@�ff�B=q@�33A���C��                                    By32�  �          Ar�H@��O�@33@���C�33@��A��@�33A�
=C��                                    By3Ad  �          Arff@���Rff@\)AQ�C�Ff@���C�@��HA��C�+�                                    By3P
  
Z          Ax  @�\�W�@z�@�C�Ff@�\�IG�@�  A��RC��                                    By3^�  
�          A|z�@��a�?�ff@���C�)@��Tz�@��A�
=C�Ǯ                                    By3mV  �          At��@��H�T  ?�{@�=qC�}q@��H�G\)@���A�{C�=q                                    By3{�  
�          Ao�@�33�P��?�{@���C�G�@�33�C33@�\)A�\)C��                                    By3��  �          Aq�@����U��?��@ƸRC���@����H��@��HA�\)C�e                                    By3�H  "          Av=q@�(��X(�?���@���C��@�(��Jff@��A��C��{                                    By3��  
(          Ao�
@�33�Q�?���@�\)C�AH@�33�C�@�
=A���C�3                                    By3��  
�          Ao
=@��
�PQ�?�@�C�U�@��
�C\)@�=qA���C�                                      By3�:  �          Am��@��
�P��?��
@�z�C��H@��
�C\)@�{A�p�C��                                    By3��  T          Amp�@أ��R=q?��\@�C��H@أ��F�\@�\)A��RC�L�                                    By3�  T          Ao�@���V�\?z�H@qG�C��@���L  @���A���C���                                    By3�,  
          ApQ�@���P��?�p�@�z�C���@���D(�@�p�A��HC�ff                                    By3��  3          AtQ�@��
�T��?�Q�@�
=C�z�@��
�I��@�
=A�=qC�(�                                    By4x  	`          Aq@���S�?J=q@?\)C�ff@���I@�=qAyC��)                                    By4  	.          Am@����M�?�p�@��RC��q@����Ap�@���A�{C��                                    By4+�  "          Ak�@�Q��N�\?s33@n{C�AH@�Q��D(�@�p�A��C���                                    By4:j  �          AmG�@�\)�Qp�>L��?G�C�f@�\)�I�@_\)AY�C�w
                                    By4I  T          Aj�R@����Q녾�p���
=C�<)@����L��@=p�A:=qC���                                    By4W�  
�          Ag
=@����P(��}p��|��C���@����Mp�@�A��C��=                                    By4f\  "          AlQ�@�ff�T�ÿ�p�����C�}q@�ff�T  ?�Q�@�=qC���                                    By4u  "          Ak33@�Q��R�H�޸R��=qC��{@�Q��S
=?�@У�C���                                    By4��  "          Aj�H@��H�U��   ���\C��{@��H�V=q?�(�@�\)C��f                                    By4�N  "          Ak
=@�(��U녿�����z�C��)@�(��Tz�@A
=C��\                                    By4��  
�          AhQ�@Å�R=q?!G�@ ��C�w
@Å�H��@\)A33C��q                                    By4��  
�          Af�H@޸R�Ip��z���\C�y�@޸R�D��@+�A+�
C���                                    By4�@  "          Ah��@��E�?h��@fffC��3@��;�@�=qA�Q�C�ff                                    By4��            Al  A  �B=q?^�R@X��C�RA  �8  @~�RA{\)C���                                    By4ی  "          AmG�@�{�G
=���\��C�E@�{�Ep�@   @�  C�^�                                    By4�2  �          Am�@�\�E�<���7�C��H@�\�K33>�p�?�
=C�h�                                    By4��  T          Ak33Az��:�R��\��C��Az��6ff@#33A"ffC�c�                                    By5~  "          Ab�HA�$  @5A9�C�1�A�ff@�p�A�  C�Ф                                    By5$  T          Ac33Ap��%p�@,(�A/33C�
=Ap��Q�@��A�C��R                                    By5$�  T          Ad��A�\�-�?ٙ�@�33C��A�\�   @��A�z�C�@                                     By53p  �          Ab�HA
=�8  �������C���A
=�6=q?�Q�@�(�C���                                    By5B  �          A`��A�4��=��
>��
C�C�A�-�@C�
AH��C���                                    By5P�  �          Ac�A��6�\�#�
�#�
C�O\A��0(�@@��AC�C��                                    By5_b  T          Ae�A  �;��}p��|(�C���A  �8��@
=qA
=qC��H                                    By5n  �          Af=qA
�R�7�
������
C��A
�R�2=q@333A3�
C���                                    By5|�  "          Ac�A(��6=q�!G��#33C�c�A(��2=q@�A�C��\                                    By5�T  
Z          A_\)A���,��>\)?�C��)A���%�@AG�AHQ�C�.                                    By5��  
�          A^{AG��+33���
��=qC���AG��&=q@#�
A)�C�+�                                    By5��  "          A]A���.{�+��0��C��A���*�\@�A�C�^�                                    By5�F  "          A]p�A
=q�,z�333�:=qC�XRA
=q�)�@�RAQ�C���                                    By5��  "          A^�HA���/\)�h���n�RC���A���,��@A	C�33                                    By5Ԓ  
�          A_
=A (��5p���
=��(�C���A (��3�
?�
=@��RC��q                                    By5�8  T          A^�RA
=�-����
��\)C�` A
=�+
=?���A ��C��=                                    By5��  
�          A\��A\)�*�R���
����C��qA\)�(��?�z�@�ffC��f                                    By6 �  T          A\Q�A(��(�Ϳ�ff��p�C��RA(��((�?�\)@أ�C��                                    By6*  
�          A[�A(��'\)��z���z�C��RA(��'
=?�  @ə�C��q                                    By6�  T          AZ�\A\)�+���
�
�\C��{A\)�.{?u@���C���                                    By6,v  
Z          AZ{@�z��/\)�.{�8z�C�s3@�z��4z�>Ǯ?У�C�\                                    By6;  
�          AZ=q@��
�-��Fff�R{C���@��
�4��<#�
=uC���                                    By6I�  T          AZ=qA�H�#��n{�y��C���A�H�!p�?�
=A�\C���                                    By6Xh  "          AX(�@��
�*=q�\)�p�C�>�@��
�-��?J=q@X��C��)                                    By6g  
�          AY��@�33�-��� Q�C��@�33�1p�?:�H@FffC���                                    By6u�  
�          A\��@��0Q��!��(z�C��q@��4��?#�
@(��C��=                                    By6�Z  
�          A[�@�  �/��.�R�7�
C���@�  �4��>�(�?�C�4{                                    By6�   "          AZff@���/
=�2�\�<��C�� @���4z�>�p�?���C��                                    By6��            A[
=Ap��-녿����p�C�S3Ap��/�?�p�@��C�5�                                    By6�L  3          AYG�@���,���ff�33C��@���0Q�?E�@P��C��
                                    By6��  T          AX��@����,  �1G��<��C��@����1p�>�Q�?��C���                                    By6͘  
Z          AY�@����-��5��?�
C��R@����2�H>���?�33C�g�                                    By6�>  
�          AW�@��\�+33���  C�
@��\�.�\?W
=@c�
C��3                                    By6��  C          AV{A33�&�R�����C�RA33�'\)?���@���C��                                    By6��            AUA��%녿��H���
C�7
A��&�H?�ff@��\C�"�                                    By70  T          AVffA(�� zῚ�H��C��\A(��\)?�Q�@���C���                                    By7�  
�          AW�
A(��%��*�H�6�RC�S3A(��*�\>\?�\)C��                                    By7%|  
9          AV�HA��&{�(��=qC�1�A��)�?^�R@l��C��                                    By74"  
�          AV�RA���'��
=���C��
A���*ff?z�H@�C��                                     By7B�  "          AW�A��'�
�����C�\A��)p�?���@�33C��                                    By7Qn  
�          AV=qA�R�'33��{�   C�  A�R�(��?�(�@�
=C��                                     By7`  "          AT��A ���&�R������(�C��{A ���(  ?�G�@�C��R                                    By7n�  T          AS
=A�#�
������ffC�4{A�%G�?��H@�  C�{                                    By7}`  �          AS\)A=q�#�
��=q��\)C�@ A=q�%G�?��H@��C�                                      By7�  "          AT��A�H�$�ÿ�=q��C�8RA�H�&ff?�p�@��\C��                                    By7��  C          AW\)A33�(  ��ff��ffC��)A33�)�?�=q@�p�C���                                    By7�R  
�          AY��A(��&�H�0���;\)C�.A(��,z�>���?�
=C��R                                    By7��  
�          A^ffA	��(���8Q��?�C���A	��.�H>��
?�=qC�                                    By7ƞ  �          A`  A
=�)G��8Q��=�C��3A
=�/\)>�{?�33C�4{                                    By7�D  
�          A`(�A���&ff�O\)�U�C�)A���.ff�L�;L��C�u�                                    By7��  
Z          A`��A�R�%G��Tz��Z�RC�g�A�R�-���\)��C��
                                    By7�  �          Ad(�A���#�
��33����C��qA���/�
�h���h��C��q                                    By86  "          Ac33A�\� �����
���C��A�\�/��������C���                                    By8�  T          Ad  A���
=��\)��
=C�%A���.=q��=q�˅C��R                                    By8�  �          AeAQ��"=q������z�C�8RAQ��.�\�u�uC�,�                                    By8-(  T          Aep�Az��$���l(��n�RC�fAz��.�R��׿��C�.                                    By8;�  
�          A]��A����������ffC��A���)p������ffC��                                    By8Jt  
�          A\Q�A���H�����p�C��A��(�ÿ�����Q�C��=                                    By8Y  "          A[\)A
=��R�s33����C�aHA
=�%�=p��G
=C�c�                                    By8g�  
�          A]A(���\�����G�C�RA(��)�Ǯ��ffC�                                    By8vf  �          Ad(�Ap��"�R��z����C��Ap��/
=�fff�hQ�C���                                    By8�  T          Ag33A���$���{��|��C�qA���0  �(���'
=C�,�                                    By8��  	�          Ah(�A��((��O\)�N�\C��A��0  =�\)>�\)C�^�                                    By8�X  �          Af{A33�'
=�<(��<��C�3A33�-p�>�{?�{C��=                                    By8��  
Z          Ag33A��&�R�-p��,��C�Z�A��,(�?�@G�C���                                    By8��  
�          Ag\)A��'
=���C�|)A��*�\?n{@l(�C�/\                                    By8�J  
�          Ag\)A���'
=� ��� (�C��
A���(��?�  @�ffC�l�                                    By8��  
�          Ah��A���)p���33��Q�C�b�A���*�\?�z�@���C�G�                                    By8�  T          AhQ�A�'������RC��HA�(��?�33@���C���                                    By8�<  T          Ag�Ap��'���=q���C���Ap��(z�?���@��C���                                    By9�  �          Af�RA��$�ÿ�Q���
=C��HA��&�\?��@�
=C���                                    By9�  "          A_\)A��%��]p��ep�C��A��.{����C�^�                                    By9&.  �          A[\)A�����ff����C�c�A�� ��?���@��C�K�                                    By94�  �          A\��A����(��"�RC�^�A��#�
?@  @G
=C���                                    By9Cz  "          A^{A����~{����C���A��'\)�G��N�RC�u�                                    By9R             A\z�A������S�
�^�\C�o\A���%������C��=                                    By9`�  �          AZ�HA���R�U��aG�C��{A��#��.{�8Q�C���                                    By9ol  T          A\z�A����
�Y���d��C��fA���$�þW
=�aG�C���                                    By9~  
�          AZ�HA�\�33�j=q�xz�C�J=A�\�%��׿��HC�Z�                                    By9��  
�          A[
=A(��\)���
���
C��\A(��$Q�xQ����C���                                    By9�^  
�          AM��A  �\)��=q�33C��HA  ��?���@��C��{                                    By9�  
�          ALQ�A33��H��33��{C��A33��?�\)@�C��q                                    By9��  �          AK�A	���  ��z��˅C�u�A	����?Ǯ@�G�C�~�                                    By9�P  �          AH��A�R�  �fff��(�C�(�A�R���@�\AC�q�                                    By9��  
�          AF{A��녿c�
���
C�0�A��
=@G�A=qC�y�                                    By9�  3          ADQ�A���׿�=q���C�*=A���R?�A  C�Z�                                    By9�B  T          AC�Az����ff��(�C���Az���?Ǯ@�\C��R                                    By:�  "          AJ{A��  �0���J�\C�s3A���\>aG�?�G�C�Ф                                    By:�  �          AP��AQ��G���  ����C��AQ���
?�G�@�  C�'�                                    By:4  
Z          AU�A����Ϳz��\)C��A����
@{A+
=C��3                                    By:-�  �          AS�A������H�ffC�ФA���@$z�A3�C�T{                                    By:<�  6          AS�A���ff��\)�=qC�
=A���(�?��H@���C��                                     By:K&  T          AS
=Az��Q��"�\�1p�C��Az��p�?��@Q�C��                                     By:Y�  �          AT��A=q�ff�z��33C�S3A=q���?�\)@��HC�R                                    By:hr  �          Aa�A  �z��!��&{C�u�A  �!�?:�H@>{C��                                    By:w  
�          A\z�A���{�{�Q�C�s3A���G�?z�H@��\C�&f                                    By:��  
�          AX��A33�{��ff�љ�C�<)A33�?�\)@�33C�AH                                    By:�d  �          AV=qA��{��33��  C���A���?�G�@��C���                                    By:�
  �          AU��AG���R������C�]qAG��?�p�@�{C�t{                                    By:��  
�          AX(�A���p��ٙ���C�"�A���{?�p�@�G�C�{                                    By:�V  T          A[\)Ap��G��z��
=C��Ap���?��H@�=qC���                                    By:��  T          AX��A33�\)���(�C���A33��?���@���C�}q                                    By:ݢ  �          AZ�HA�\�p���33���C���A�\�z�?��
@�C��3                                    By:�H  "          A[�
A��p��\�˅C���A����?�@�\)C��)                                    By:��  �          A]p�A
=��������RC��A
=�!p�?p��@z=qC���                                    By;	�  �          Ab�HA
=�!���  �{C��A
=�$z�?�z�@�{C��f                                    By;:  
�          Ac
=A�H�!p�����
=C��A�H�%p�?xQ�@{�C��\                                    By;&�  �          AaG�AQ���
��\��ffC�.AQ�� Q�?˅@�  C�"�                                    By;5�  �          Aap�A\)���	�����C�qA\)�!�?�p�@�  C��                                    By;D,  T          A`��A=q�33����C�
=A=q�"�H?��\@���C��
                                    By;R�  
�          A`��A����=q�C��)A��#\)?�  @�=qC��                                    By;ax  �          A`z�A��=q��	p�C�FfA�� Q�?��\@�{C�3                                    By;p  T          A^�\A����	���ffC�� A��=q?�@��C�AH                                    By;~�  T          A\  AQ�������� ��C�b�AQ����?fff@p  C��                                    By;�j  
�          AYG�Aff�=q��33��\)C��Aff��\?��@�G�C��)                                    By;�  
�          AV{A��  ��{� (�C��A��G�?���@�ffC���                                    By;��  
�          AVffAz��Q�޸R��{C�!HAz����?\@ϮC�3                                    By;�\  �          AV�\Ap���׿�(���\)C��{Ap���R?�
=AQ�C�%                                    By;�  T          AQ�A�
�����\���RC�L�A�
���@G�A�C��3                                    By;֨  �          AM�A��
�\�h����=qC��=A���@G�Az�C�)                                    By;�N  
�          AP��A�(����
��Q�C��{A�	��?���A	C�
                                    By;��  �          AP��Ap���׿�����ffC���Ap��
�\?�\)A�
C��{                                    By<�  "          AO33A���녿��R��Q�C�B�A���Q�?��@��RC�j=                                    By<@  �          AN�\A�R�  ���R����C���A�R�=q?�A33C��                                    By<�  �          AO�A��Q쿪=q��z�C��{A��
=?�\@��HC��3                                    By<.�  �          AR{A33�
=����{C���A33�Q�?���@���C�c�                                    By<=2  "          AR�HA=q���� ���(�C�EA=q��R?�p�@��C��                                    By<K�  
�          AP  Aff����Ǯ���C��{Aff�p�?˅@���C��
                                    By<Z~  C          AK�A=q�����H�ҏ\C�]qA=q�
=?�33@�
=C�k�                                    By<i$  
�          AJ=qA{����
��{C��=A{���?�ff@��C���                                    By<w�  �          AK�
A
=��R�ٙ�����C��=A
=�\)?�@�z�C�u�                                    By<�p  �          AK�Az��G���G��ٙ�C��\Az���?�=q@�C��{                                    By<�  
Z          AJ=qA���׿�\)�ƸRC���A���?�Q�@�ffC��                                    By<��  
�          AM�A
=�����p�����C�A
=��
?�AQ�C�9�                                    By<�b  "          AN�RA
=�33����ڏ\C��HA
=�
=?���@�C��                                    By<�  "          AM��A��{��G����
C��A��
=?�\)@��
C��f                                    By<Ϯ  
<          AMG�A���{��p�����C��RA���
=?�33@�Q�C��                                     By<�T  "          AL(�Ap��(���p���\C��Ap���\?���@��\C��=                                    By<��  "          AM��A33�녿�  ���\C�8RA33�  ?���A�C�j=                                    By<��  
�          AN�RA���H��Q���z�C�*=A���?��@��C�C�                                    By=
F  �          AM�A����R���H��G�C�!HA�����?�{Ap�C�<)                                    By=�  
Z          AMG�Ap���\�\�ڏ\C��qAp��{?��H@�C�
=                                    By='�  �          AL��Az���H�У���G�C��qAz���H?У�@��C�޸                                    By=68  �          AMG�A�
��
������HC��3A�
��
?��@��
C��3                                    By=D�  
�          AMAQ��  ��Q���G�C��)AQ��Q�?�{@�C���                                    By=S�  �          AL��A���p�����33C�=qA����\?��H@�G�C�                                      By=b*  "          AJ�\A(���H����=qC�h�A(���?��R@���C�T{                                    By=p�  �          ALz�A
{������\)C��fA
{���?�
=@�{C�ff                                    By=v  
�          AJ�RA  �G���R�"=qC��)A  �z�?��@�{C�q�                                    By=�  
(          AJ�\A�\��
��
�{C�'�A�\�{?��\@�Q�C��\                                    By=��  �          ALz�A�R����R� Q�C���A�R���?�@��C���                                    By=�h  
I          AL(�A�R�G��
�H���C��A�R�  ?��H@��RC���                                    By=�  e          AL��A
�R�33�z����C��A
�R���?�G�@��C�k�                                    By=ȴ  T          ALz�A(�����G����C�7
A(���H?���@\C��                                    By=�Z  "          AMG�A33��
��=q��C���A33���?�G�@�Q�C���                                    By=�   
�          AL��A�
�33������HC�� A�
�33?�Q�@�=qC���                                    By=��  
�          AK�
A�������H�C�l�A���G�?��@���C�C�                                    By>L  
(          AL��A�R��33�%p�C��qA�R��?���@��C���                                    By>�  
�          AL(�A
�R�ff���R��
C���A
�R�Q�?��@�G�C���                                    By> �  T          AL(�A  �Q��Q��{C�C�A  ��H?�G�@��C�f                                    By>/>  T          AL��A
�R�
=��\�
=C���A
�R�G�?���@�p�C�t{                                    By>=�  T          AL��A������������C�5�A���ff?���@���C�!H                                    By>L�  �          AM�A ���p����H��p�C��qA �����?�\)A=qC���                                    By>[0  T          AK\)AQ���׿�  ��G�C�u�AQ��33?��HA�C��{                                    By>i�  "          AK33Aff��H�����љ�C�ٚAff�G�?�p�A
=C���                                    By>x|  T          AJ�HAff�=q����\C���Aff�(�@z�A=qC�)                                    By>�"  T          AL��A�������33���C�˅A���33?�{@��HC���                                    By>��  T          ALQ�A����H�33�&ffC���A���33?W
=@p��C�w
                                    By>�n  T          AM��A�H�� ���4z�C�8RA�H�\)?�R@0��C���                                    By>�  T          AN�\A�H�����-C��A�H���?=p�@Q�C���                                    By>��  �          AL��Aff�����H�.=qC�0�Aff�
�\?5@K�C��=                                    By>�`  "          ALQ�A(��p��&ff�<Q�C��A(���
>�ff@�C�                                      By>�  �          AK�Ap����*�H�AC�P�Ap��
{>�G�?�p�C���                                    By>��  "          AK�A�R�  �  �"�HC�ffA�R�(�?Y��@s�
C��
                                    By>�R  
�          AK�A��  �=q�/
=C�S3A��	�?0��@HQ�C���                                    By?
�  
�          AL(�A��� z���R�333C��A���=q?\)@ ��C�n                                    By?�  
{          ALQ�Az�����$z��9��C��3Az���?   @��C�+�                                    By?(D  
          AK�A�� z��{�2�RC�HA��{?z�@%C�ff                                    By?6�  
�          AK
=A�� ������2=qC��\A��ff?(�@/\)C�8R                                    By?E�  �          AI�A�
�  ����/
=C�qA�
���?:�H@Tz�C��
                                    By?T6  
�          AI��A
=�Q��Q��.=qC��A
=�	�?@  @\(�C��                                     By?b�  
Z          AJ�\A{�{�{�3�
C���A{��?!G�@5C��3                                    By?q�  
�          ALQ�A�����'
=�<z�C���A���\)>�@��C�:�                                    By?�(  e          AMp�A\)���R�*�H�@Q�C�T{A\)�{>\?�Q�C��
                                    By?��  T          AMA�\��  �)���>ffC���A�\�
=>���?��HC�8R                                    By?�t  	�          AM��A�H���R�'��<��C�RA�H�=q>�{?\C�W
                                    By?�  
�          AL��A����  �!��6�\C��fA���=q>�ff@   C�7
                                    By?��  �          AMp�A{����,���BffC��{A{�33>�z�?��\C�(�                                    By?�f  �          AMp�A �������,(��Ap�C���A ���   >aG�?�  C���                                    By?�  	�          AP��A%����,���?�C�1�A%�����>.{?E�C�XR                                    By?�  �          AJ�RA$z����H�8Q��RffC�.A$z���p��L�Ϳh��C�
                                    By?�X  
Z          AJ�\A#
=�޸R�5��N�RC��3A#
=��׽��;��C�˅                                    By@�  
�          AHz�Aff��\�
=�.ffC��Aff��p�?\)@#�
C�N                                    By@�  T          AEAQ�����(���HC��{AQ���?�z�@��
C���                                    By@!J  �          AEA33����*=q�G
=C��
A33��
>�p�?��HC�3                                    By@/�  T          AF�RA�R������,��C�J=A�R� ��?(��@C33C���                                    By@>�  
�          AEG�A=q�����(��6�\C��{A=q��
?�R@7�C���                                    By@M<  
�          AB{AG������\�.=qC��AG��Q�?J=q@n�RC�p�                                    By@[�  
�          ABffA���������C���A�����?��
@��C�XR                                    By@j�  �          AB{A
ff�z��Q���HC�!HA
ff��\?�p�@��\C��f                                    By@y.  �          AA�A	���\�ٙ���C��A	��33?�  @�C��{                                    By@��  
�          AAG�A	�����˅���HC�޸A	����?�=q@���C��q                                    By@�z  T          A@��A	G��Q��(��\)C��A	G���?�Q�@ۅC���                                    By@�   T          A?�
A����z��
=C��A���?��@�ffC�Q�                                    By@��  �          A@Q�A\)�����ff�	�C��\A\)��H?�33@�p�C��                                    By@�l  �          A@��Ap��	녿�\)���HC��Ap���H@Q�A#33C�W
                                    By@�  
{          A?�AG��zῐ������C�(�AG����@ffA!G�C�y�                                    By@߸            A>ffAQ��33��{��=qC�4{AQ����?���A
=C�]q                                    By@�^  T          A=A�
�=q���
��C�:�A�
��?�AC�Ff                                    By@�  
I          A<��A=q��ٙ��z�C��)A=q��\?�@��C��                                    ByA�  e          A>=qA
=��\��=q��C��)A
=�(�?��@�33C��\                                    ByAP  
�          A@(�A	p���H��{��\C�1�A	p����?�ff@�
=C��                                    ByA(�  T          A@��A�����Q��z�C��)A��\)?�Q�@���C�Z�                                    ByA7�  T          A>�\A��G��
=q�'
=C�1�A����?}p�@���C���                                    ByAFB  �          A<z�A����\)����@z�C�\A����?.{@U�C�u�                                    ByAT�  
�          A;33A
=q��=q�G��333C�ffA
=q���
?333@Z�HC��
                                    ByAc�  
�          A<(�AG����z��"ffC��RAG���
=?\(�@�{C�h�                                    ByAr4  T          A>=qA������\)�Ap�C��A�����R>��?��HC��                                    ByA��  
{          A@(�A�H��  �.�R�RffC��RA�H��  >#�
?E�C��=                                    ByA��  3          A>�\A(��У��<���eC���A(����;��R��  C���                                    ByA�&  �          A=�A33����=q�<��C�eA33��>��H@ffC���                                    ByA��  u          A:{A�������,��C��3A���{?=p�@hQ�C�L�                                    ByA�r  
�          A7�A�����9���j{C�"�A��ٙ��Ǯ��Q�C��\                                    ByA�  3          A;
=A���ff�:�H�g�C�g�A����H���Ϳ���C�{                                    ByAؾ  
�          A9�A33��\)�;��j�RC���A33��z���Q�C�s3                                    ByA�d  �          A5G�A  ����7
=�i�C��{A  ��  ����
C�t{                                    ByA�
  T          A6{AG��˅�!��N{C�� AG����H<�>��C��                                     ByB�  "          A7
=Az���z���<z�C��{Az�����>��
?���C��                                    ByBV  T          A6ffA�R��p��=q�C
=C��)A�R��\>�=q?�{C��f                                    ByB!�  �          A9A����\)�"�\�I�C��=A���޸R=��
>���C��3                                    ByB0�  �          A:ffAp��У���H�@(�C���Ap���ff>L��?}p�C���                                    ByB?H  "          A5�Aff��p����((�C�y�Aff��\)>��H@p�C���                                    ByBM�  T          A-A�
�љ�����C�,�A�
��ff?h��@�=qC��q                                    ByB\�  �          A$��A��������p�C�S3A����Q�?�G�@���C�&f                                    ByBk:  T          A ��@�������&�\C��f@��Å?�@@��C�1�                                    ByBy�  �          A Q�@�����Ϳ�\)���
C�Ǯ@�����?s33@�{C���                                    ByB��  "          A"�\@��
�ȣ׿��
���C���@��
�ə�?�
=@���C��=                                    ByB�,  
�          A(��AG��ָR�Tz�����C�"�AG�����?�G�A{C��f                                    ByB��  
k          A)p�A(����ÿ�G����C��{A(���?��
A��C��                                    ByB�x  �          A+\)A33�ٙ����
����C�,�A33��{?�{A
=qC�g�                                    ByB�  T          A.�\A
=���
�����  C�}qA
=��Q�?�
=A��C��R                                    ByB��  "          A/�A�H���
��ff�p�C�h�A�H��{?�z�@��HC�B�                                    ByB�j  �          A/33A  ��33���R�(  C�{A  �ۅ?&ff@Z=qC���                                    ByB�  
�          A1�A
{��=q�\)�:ffC�]qA
{��>�
=@	��C��                                     ByB��  
�          A0  A����z���R�&�HC��A������?+�@`  C���                                    ByC\  
�          A0z�A\)�����(��7\)C�ٚA\)��  >���@�
C�)                                    ByC  �          A0��A�����\�=q�I��C��)A���ə����
����C���                                    ByC)�  T          A/\)A��
=�*�H�a��C���A��=q�����z�C�'�                                    ByC8N  "          A(��@�z���\)��
=�(Q�C�@�z��޸R?E�@�ffC�J=                                    ByCF�  �          A*ff@����p����M��C��@���陚>�G�@ffC�U�                                    ByCU�  
�          A+33@��
��=q�!��Z�HC��3@��
��  >�{?���C���                                    ByCd@  C          A+�@�����*=q�e�C�*=@������>��?�z�C�@                                     ByCr�  
�          A)�@�=q��33���1��C�q�@�=q��=q?h��@�ffC�                                    ByC��  �          A$z�@�(���=q�n{��
=C��@�(���@G�A5�C���                                    ByC�2  �          A'\)@�=q��
=�{�[33C�7
@�=q��p�>.{?s33C�AH                                    ByC��  @          A$Q�@�
=��z�8Q�����C��\@�
=���H@\)AK
=C�AH                                    ByC�~  �          A#
=@�G����
�s33��p�C�S3@�G����@G�A8��C���                                    ByC�$  �          A#
=@�z���p����%C�s3@�z���  @%Ak�
C�8R                                    ByC��  �          A!@����p��Y�����\C�AH@����{@�\A:�\C��R                                    ByC�p  
�          A&ffA����
=��(�����C�'�A����\)?�33@�p�C�                                      ByC�  
�          A)p�A�H��
=��ff���C���A�H����>k�?�  C��                                    ByC��  
�          A(Q�A\)��Q��{�G�C���A\)��
=?�@3�
C�J=                                    ByDb  T          A'�
A���
=��{���C�aHA����R?�z�@˅C�g�                                    ByD  
Z          A*=qA����33��p���(�C�1�A����(�?���@�C�#�                                    ByD"�  
�          A)Aff��Q��
=�=qC�qAff���?��@>�RC��R                                    ByD1T  �          A(z�Az����������z�C��
Az���  ?h��@�\)C�c�                                    ByD?�  �          A&{A
=���\���
��{C���A
=����?�
=@љ�C���                                    ByDN�  �          A%�A33��33�!G��]p�C�� A33���?ǮA
�HC��                                    ByD]F  �          A%p�A	���{���&ffC�EA	�����@z�A9G�C�0�                                    ByDk�  
�          A%��A�����
<�>\)C�K�A����ff@�A:ffC�O\                                    ByDz�  �          A$z�Aff���׿O\)����C�L�Aff��{?�@ҏ\C�z�                                    ByD�8  �          A&ffA  ���\�5�{�C�H�A  ���R?��@���C���                                    ByD��  "          A(z�A����=q��{���C���A�����\>��?�
=C�S3                                    ByD��  T          A(��A�����׿�{�"�HC��
A�����8Q�xQ�C���                                    ByD�*  
�          A&�RAff��
=���
��C�XRAff��ff>���@�C��=                                    ByD��  �          A'33A���\�\���C�|)A��  ?#�
@a�C�3                                    ByD�v  
�          A-��A����33������C��A�����?z�@C�
C���                                    ByD�  �          A/\)A�����
�޸R�{C��fA����z�>�Q�?�\)C�C�                                    ByD��  �          A.�\AG���p���(����C�h�AG����R>���?���C���                                    ByD�h  T          A-�A  ��G�������C��A  ��  >�@#33C���                                    ByE  
�          A-��AQ����R��=q��G�C��HAQ�����?^�R@��C���                                    ByE�  �          A+33A����=q��(���G�C�0�A�����?��\@��RC�R                                    ByE*Z  �          A'
=A
=����������C��RA
=��{?h��@���C�e                                    ByE9   �          A#33A����p�������C���A����Q�?^�R@��
C���                                    ByEG�  �          A"�\A����ff��{�C�fA����ff>��
?�=qC�b�                                    ByEVL  T          A&�\A������  �G�C�9�A�������\�4z�C��{                                    ByEd�  
�          A&ffA\)��  �G��J=qC���A\)�������"�\C�9�                                    ByEs�  
Z          A)G�A����\����P��C�W
A���{�!G��[�C��{                                    ByE�>  
�          A2ffA���  ���<��C��A���Q쾏\)��C��=                                    ByE��  
�          A4��A=q���R�p��4��C���A=q�����.{C��                                     ByE��  
�          A3�
A���	���0��C��
A��(���Q��G�C���                                    ByE�0  �          A5A
=��G��Q��,��C��A
=��
=���
��Q�C���                                    ByE��  
�          A6�HA   ��z��z��'
=C�c�A   ��G�=�\)>�Q�C��                                     ByE�|  �          A8Q�A33��
=��(����C��qA33����>�z�?�
=C��H                                    ByE�"  
�          A9A�\��녿�
=�p�C��A�\��Q�?0��@X��C�`                                     ByE��  �          A8��A(���  ��Q���p�C�8RA(����H?z�H@�z�C�f                                    ByE�n  T          A:�RA��Q��ff����C�S3A��z�?c�
@��C��                                    ByF  �          A;\)A33����Ǯ��(�C�z�A33���
?^�R@���C�1�                                    ByF�  �          A8��A����
��Q���z�C��=A��Å?��\@ə�C��3                                    ByF#`  
�          A7�
AG����ÿL���~{C�c�AG���33?�Q�A33C��                                    ByF2  "          A6�\AQ���  �!G��I��C�Y�AQ���Q�?�A(�C��q                                    ByF@�  
�          A6�\AG���G��&ff�P��C�~�AG�����?�
=A\)C��                                    ByFOR  �          A7�A33��(������
=C���A33�ʏ\?�z�@�=qC�R                                    ByF]�  
�          A8Q�A����33��Q���ffC��{A����ff?n{@�z�C�\)                                    ByFl�  �          A7�A(���녿�=q��{C���A(���
=?G�@z�HC�E                                    ByF{D  
�          A8��A Q���Q����6ffC�#�A Q������\)��Q�C��                                    ByF��  �          A9G�A"�\���%��MC��A"�\��G�����ffC���                                    ByF��  T          A9�A Q���
=����0��C�9�A Q���p��#�
��\)C�=q                                    ByF�6  T          A7�
A�R���������{C��qA�R�Å?��@���C�ٚ                                    ByF��  
�          A7\)A�R���R�����
C�,�A�R����?}p�@�\)C��q                                    ByFĂ  
�          A6ffA33���׿�(��33C��A33����>�ff@  C�z�                                    ByF�(  
Z          A5A���녿���C���A����>B�\?s33C���                                    ByF��  "          A5��Az����H��Q����C�!HAz����R?Tz�@�
=C��)                                    ByF�t  
Z          A4��A���\��G���p�C�^�A���
?��@���C�K�                                    ByF�  �          A1p�A{��ff�������
C���A{���?���@߮C�                                      ByG�  �          A1G�AG������{��\C�~�AG��ȣ�@��A<z�C�O\                                    ByGf  �          A0(�AG����?�Q�@�ffC��)AG�����@tz�A���C�                                      ByG+  �          A/\)AQ���\)?u@���C��AQ����@X��A�Q�C��{                                    ByG9�  
�          A,z�A�R��  ?�G�Ap�C�*=A�R���@z=qA��\C���                                    ByGHX  
�          A.�RA  ����=�?#�
C�HA  ����@ffAF�RC�"�                                    ByGV�  �          A+\)AQ���
=?�  @��C��qAQ����
@N�RA�\)C���                                    ByGe�  �          A+�
A������>���@�C���A����p�@&ffA`  C�!H                                    ByGtJ  �          A)�A�����
=�\)>\C�4{A�����
@�AUC�Y�                                    ByG��  �          A+\)A���R���A�C�fA��
=?��
A��C��3                                    ByG��  �          A-p�A����{�Ǯ�z�C�^�A����(�?�
=A$z�C��                                    ByG�<  �          A/�AG����
�c�
��(�C�HAG����?�ffA��C�K�                                    ByG��  �          A,��A���{����C�z�A���
=@�AK
=C���                                    ByG��  �          A,z�A	���p�?aG�@�{C��{A	����\@R�\A��RC�u�                                    ByG�.  �          A+�
A
{��(�>��
?�Q�C�� A
{����@0��An=qC��                                    ByG��  �          A-�A  ��\)?�\@+�C�/\A  ��=q@1�An=qC���                                    ByG�z  �          A((�A33���?��@���C�L�A33����@Z�HA�(�C��
                                    ByG�   �          A)��A33��Q�?��@�ffC��)A33���\@^{A�G�C��                                     ByH�  �          A)��@��H��\@ ��A/33C�� @��H��Q�@��HA�C�k�                                    ByHl  �          A�
@�G��أ�@0��A��\C��@�G���@��A�33C�Y�                                    ByH$  �          A\)@����  ?У�A
=C��
@�����
@���AîC�K�                                    ByH2�  �          AG�@����ƸR?��@���C��f@�����\)@eA��\C��                                    ByHA^  �          AG�@�ff���
?h��@��\C�S3@�ff��G�@L��A���C�Z�                                    ByHP  �          A@����33?��\@���C�4{@����
=@X��A���C�O\                                    ByH^�  T          AG�@�{�Ǯ?�z�@�C��f@�{���@^�RA�  C�Ǯ                                    ByHmP  �          A@������H?:�H@��C��f@�����33@;�A��
C��                                     ByH{�  �          A�@�ff��\)>W
=?�(�C��@�ff���@%Aw�
C�`                                     ByH��  T          A�@�\)����?^�R@��HC��@�\)��Q�@B�\A�=qC���                                    ByH�B  �          A�@�=q����>�
=@(Q�C��@�=q����@*�HA��\C��)                                    ByH��  �          A=q@�Q����>W
=?�ffC�,�@�Q�����@�Aw�C��H                                    ByH��  
�          A\)@�����Ϳ=p����C��=@������?�33A  C�޸                                    ByH�4  �          A�\@���\)�L�;��RC��)@�����@��Adz�C���                                    ByH��  �          A�R@���ff�L�;�\)C�xR@����@ffAf=qC��\                                    ByH�  �          A
=@�z���\)�.{�}p�C��@�z���=q@  AW
=C��f                                    ByH�&  �          Az�@����
��33�C���@���=q?�A2�RC��f                                    ByH��  �          A  @�{��z�?�@K�C���@�{��\)@0  A���C�q�                                    ByIr  �          A�H@��\��\)=#�
>k�C�  @��\���@33AI�C�5�                                    ByI  �          A�R@�����H>��
?���C���@�����@ffAg�C�3                                    ByI+�  |          A�RA  �s�
�(��i��C�9�A  �p  ?h��@�C�c�                                    ByI:d  �          A��AQ��w
=�J=q��(�C�nAQ��w�?@  @��
C�g�                                    ByII
  �          A�A���p�׾�
=���C���A���hQ�?��@�{C�3                                    ByIW�  �          Ap�A�c33�u���HC�^�A�h��>�@333C�)                                    ByIfV  �          A�A
�H�S�
�G���=qC��\A
�H�W
=?��@X��C�l�                                    ByIt�  �          A�A
=�Vff�O\)���\C�,�A
=�Y��?
=q@Z�HC�                                    ByI��  �          A�\A��XQ�G����C�qA��Z�H?z�@j=qC�                                      ByI�H  �          A�
A
�R�G
=�!G��z=qC�&fA
�R�G
=?(�@s33C�"�                                    ByI��  �          A�RA
=�6ff�333��33C���A
=�9��>�ff@4z�C�Ǯ                                    ByI��  �          A��A
=q�2�\�
=q�Z=qC�\A
=q�1�?�@fffC�3                                    ByI�:  �          Az�AG��1녾��?\)C�AHAG��0��?�R@uC�U�                                    ByI��  �          A��A
=�   �\)�]p�C�1�A
=�!G�>�G�@-p�C��                                    ByIۆ  �          A=qA����ÿz��aG�C���A����>Ǯ@ffC�|)                                    ByI�,  �          A�\A=q��\���dz�C���A=q�ff>��?���C�Z�                                    ByI��  �          A\)A�Ϳ��=p����
C���A�Ϳ��;\)�^�RC���                                    ByJx  �          A{A�
��(��
=�k�C�XRA�
�˅<��
>\)C���                                    ByJ  T          A�A녿���Ǯ�{C�NA녿�ff>��
@33C�B�                                    ByJ$�  T          A(�A����;�  ��=qC�
A���ff>�@B�\C�:�                                    ByJ3j  �          A�
AG����
��Q���C�3AG�����?   @L��C�T{                                    ByJB  �          AG�AQ�}p�>�z�?�\)C���AQ�L��?#�
@�{C�b�                                    ByJP�  �          AQ�A��aG��u��=qC�)A��h��=�G�?@  C��                                    ByJ_\  �          A33A
=q��녿c�
���
C��A
=q�8Q�+���z�C��                                     ByJn  �          A
�HA	G�����=q��C�B�A	G��fff�L�����C�                                    ByJ|�  �          A33A	녿zῆff��{C�{A	녿p�׿@  ��ffC��f                                    ByJ�N  �          A
�RA	p��녿����C�)A	p��k��@  ���RC��\                                    ByJ��  �          A��A�
����fff��(�C��A�
�fff����|(�C�                                    ByJ��  �          A��A  ��\)���\��p�C�RA  �&ff�W
=��\)C���                                    ByJ�@  T          Az�A��W
=�h����C�Q�A��
=q�B�\��
=C�<)                                    ByJ��  �          A
=A
ff����333���C�&fA
ff�   ����g
=C�U�                                    ByJԌ  �          A��A�ÿ
=q���fffC�&fA�ÿ333��\)��C���                                    ByJ�2  �          A�A�þk���\)��C�8RA�þ��
�.{��Q�C��=                                    ByJ��  �          A=qA녽u��p�� ��C���A녾B�\���
�
=qC�U�                                    ByK ~  �          A	�A��<����R�ff>B�\A�ý�Q쾙��� ��C��\                                    ByK$  �          A	�A��=�\)��  �ٙ�>��HA�ü������G�C��                                    ByK�  �          A�A��>�{�W
=��@�A��>u���
�p�?�33                                    ByK,p  �          A  A  ��>8Q�?�p�C��RA  �#�
>W
=?�Q�C��)                                    ByK;  T          A  A�>���>.{?�Q�@333A�>�G�<#�
=#�
@C33                                    ByKI�  �          A\)A�H?&ff=��
?
=@���A�H?!G��.{��@��                                    ByKXb  �          A�A\)?�\    <#�
@c�
A\)>�׾B�\���@S33                                    ByKg  �          A�\Aff<#�
>�Q�@p�=L��Aff>\)>���@��?z�H                                    ByKu�  �          A��A��>#�
���
��?��A��>\)���
��?u                                    ByK�T  
�          A=qA�>�ff�#�
���@EA�>��;W
=��z�@1G�                                    ByK��  T          A=qAp�?0��>��@L(�@�ffAp�?O\)>.{?�
=@��                                    ByK��  �          A{AG�?z�?
=q@l(�@|��AG�?=p�>�\)?�
=@���                                    ByK�F  �          A�\A��?!G�?
=q@mp�@�=qA��?J=q>�=q?�@���                                    ByK��  �          AffA?+�>�ff@C33@���A?G�>#�
?�{@��                                    ByK͒  �          A�RA(�?�\)?�G�@�@�ffA(�?�?�@a�A�                                    ByK�8  �          A�HA��?�Q�?O\)@�  A�\A��?�z�>���@�A�                                    ByK��  �          A�A  ?�
=?Q�@��HA�A  ?�33>��R@Q�A                                    ByK��  �          A=qA�>�
=?��
@�=q@8��A�?G�?L��@��R@��H                                    ByL*  �          A  A�\?�?��
@�ff@w�A�\?h��?=p�@�\)@�ff                                    ByL�  �          A  Aff?!G�?���@�{@��Aff?�  ?G�@���@�G�                                    ByL%v  �          A��A�R?\)?�G�A\)@qG�A�R?}p�?s33@�p�@�                                    ByL4  
�          A�A{?��?��HA�@o\)A{?u?k�@�\)@��                                    ByLB�  
�          A(�Aff>�z�?��
A
{@   Aff?@  ?���@�  @�33                                    ByLQh  �          A�
Ap�?   ?�(�A33@\��Ap�?��\?�@�@�{                                    ByL`  �          A�
A��?�?�A5��@h��A��?�{?���A�\@���                                    ByLn�  �          AG�@��R?�p�?�p�A?\)ADz�@��R@  ?s33@��A|��                                    ByL}Z  T          A��A ��?���?�  A&�HAz�A ��?��
?fff@�\)AG�                                    ByL�   T          A��AG�?�?�ffA  A�
AG�?�?+�@���AIp�                                    ByL��  D          A��A�?Ǯ?�=q@�A/�A�?���>�
=@8��AO
=                                    ByL�L             A�HA
=?��
?fff@��
AD  A
=?�p�>.{?�z�AZ{                                    ByL��  "          A�A�?�33?L��@�ffAO�A�@�
=#�
>�=qA`��                                    ByLƘ  T          AQ�Az�?��?&ff@�z�AM��Az�@   ���Ϳ+�AX��                                    ByL�>  "          A  A  ?�
=?�R@��AR=qA  @G�����z�HA[�
                                    ByL��  �          A�R@��R@�\?��RA#33A�Q�@��R@+�?�@g�A���                                    ByL�  T          A{A�@#�
@Q�A]A�ffA�@J�H?��\@ӅA��\                                    ByM0  
(          A=qA�@,(�@��AyA��RA�@X��?�(�@�z�A��                                    ByM�  "          A
=Aff@,(�@\)A�p�A�Q�Aff@[�?��A33A�ff                                    ByM|  �          AQ�A   @'�@�HA�z�A��HA   @U�?��
A�\A��R                                    ByM-"  �          A
ffA   @!G�@ffAaG�A�(�A   @G�?��\@�Q�A��R                                    ByM;�  
�          A��A Q�@p�?ǮA((�A�ffA Q�@7
=?
=q@fffA��                                    ByMJn  �          A	�A@=q?��AG�A���A@.�R>���@��A���                                    ByMY  
�          A	A=q@=q?���A\)A��A=q@1G�>�@Dz�A�ff                                    ByMg�  
�          A
=A��@?�\)@�z�A|��A��@%>.{?��A���                                    ByMv`  
�          A
ffA�\@"�\?�  A��A�=qA�\@4z�>k�?\A���                                    ByM�  
�          A
�\@�z�@G
=?�  A:{A�(�@�z�@b�\>�@I��A��                                    ByM��  "          A
{@�  @U�?�ffA@(�A��@�  @p��>�G�@<��A���                                    ByM�R  T          A33@��\@P��?���AO\)A���@��\@p  ?�R@��Ȁ\                                    ByM��  
�          Ap�@�p�@J�H@z�As
=A�z�@�p�@s33?}p�@�p�A���                                    ByM��  �          Aff@��R@G
=@��A�A���@��R@r�\?���@�=qA�                                    ByM�D  �          A{@���@`  @
=Aw
=A�G�@���@��
?h��@��A���                                    ByM��  T          A��AG�@G�@"�\A��HA�
=AG�@u?�(�@���A�G�                                    ByM�  "          Ap�A�\@>{@*=qA�Q�A�{A�\@o\)?��A��A�
=                                    ByM�6  T          A33@�Q�@�{@G�Ar�\A��@�Q�@�\)?z�@uB\)                                    ByN�  "          Az�@�33@P��@ ��AZffA�@�33@qG�?.{@�=qA�
=                                    ByN�  �          A	��@�  @0  @
=A�{A�{@�  @Z�H?�Q�A Q�A�ff                                    ByN&(  �          A{@��@w�@z�A��RA��
@��@�ff?@  @�z�B �                                    ByN4�  �          A{@�=q@G�@ffAs�A�ff@�=q@j�H?Q�@�z�A�p�                                    ByNCt  r          @�@��H@o\)?��AQ�B��@��H@w��Ǯ�N{B��                                    ByNR  �          @�@��@�(�?333@���B�R@��@��H�p����B��                                    ByN`�  �          @��@��@`��?�A[33A�ff@��@|(�>�(�@N{A���                                    ByNof  �          A�H@��
@J=q@Dz�A�(�A�p�@��
@��?�(�A=�A�                                    ByN~  �          @��H@��R@@\)A��\A���@��R@E�?�(�AZ�HA��                                    ByN��  �          @��@mp�@ ��?�33A��
Bff@mp�@A�?\(�A#
=B��                                    ByN�X  "          @�(�?�{@mp���\�ڸRB��H?�{@.{�P  �;�\B~{                                    ByN��  
�          @�
=@c�
��@�33BYp�C�(�@c�
?(�@�  Bb�A                                      ByN��  �          A	��@�=q>���@���BMG�@Mp�@�=q@+�@��HB6�A��                                    ByN�J  "          A=q@߮<�@�=qB��>aG�@߮?�Q�@�ffB	(�AxQ�                                    ByN��  
�          A(�@�>�\)@�BQ�@z�@�@�
@�ffB��A�p�                                    ByN�  
�          A(�A   ?=p�@�G�B \)@���A   @   @�A�
=A��H                                    ByN�<  |          AffA�H@x��?��A2ffA�33A�H@���>�=q?�A�ff                                    ByO�  T          A�@���@r�\@Q�Ac\)A�@���@���?!G�@�p�A�                                      ByO�  
�          AG�@�=q>�
=@��B0z�@�(�@�=q@�H@�z�B��A���                                    ByO.  
�          @��
@�=q?�=q@��
B"�RA.�H@�=q@0  @�p�BffA�Q�                                    ByO-�  T          @�\)@���?�
=@�p�B\)Ah(�@���@@��@vffA�  A�\                                    ByO<z  T          @�\)@�z�@p�?Q�A��A�\)@�z�@
=���
�W
=A�33                                    ByOK   �          @��@�  @z�>���@dz�A�(�@�  @�\����(�A�p�                                    ByOY�  �          @��
@���?�  �����A���@���?�(�����X��A��                                    ByOhl  �          @�@�
=@�
>8Q�?��HA��R@�
=?��R�����HA���                                    ByOw  "          @��H@�p�@��>�\)@0  A�z�@�p�@ff�   ��p�A��                                    ByO��  T          @�  @�Q�?��>��@��
A��@�Q�?��H�u�&ffA��\                                    ByO�^  T          @��H@�G�?˅?:�H@�G�A��@�G�?޸R=�?��RA��
                                    ByO�  �          @��@��?���?\)@�G�AVff@��?��H=L��>�Ag�                                    ByO��  "          @У�@�33?�(�<�>�z�APQ�@�33?��׾��H���AD��                                    ByO�P  
�          @�G�@�{?��
=�?�ffA  @�{?�G���  �ffA��                                    ByO��  T          @��
@���?��
>��@G
=@���@���?�{<��
>#�
A=q                                    ByOݜ  
�          A   @��
?xQ�?�ff@�Q�@ᙚ@��
?��
?!G�@���A�
                                    ByO�B  	�          A  A��?(�?�33A2�R@�p�A��?�33?�=qA�@��                                    ByO��  �          A�R@�ff?
=?�G�AG\)@�\)@�ff?�?�Q�A"�RA{                                    ByP	�  
�          @�z�@�{>��?�ffAR�H?���@�{?aG�?˅A:ff@У�                                    ByP4  T          @��
@�G�>W
=@(�A�p�?У�@�G�?u?�(�Apz�@�Q�                                    ByP&�  "          A ��@�ff��\)@p�A�{C�� @�ff?5@Arff@���                                    ByP5�  T          A��A�H=�Q�?��
A)G�?(��A�H?�R?�33A�R@��                                    ByPD&  "          A�RAG��8Q�?�\)@��
C�XRAG�>aG�?�{@��\?��                                    ByPR�  �          A  Ap����R?���A/�
C���Ap�>�  ?�=qA0��?�ff                                    ByPar  
�          A�A   ����?��
A,(�C��A   �   ?���AM�C�7
                                    ByPp  
�          A�A Q쿵?��
A*{C���A Q�L��?�
=AW33C�'�                                    ByP~�  	�          Aff@�(���  @��A�G�C�p�@�(���@1G�A�\)C���                                    ByP�d  
�          @�  @�ff�޸R@\)A���C�5�@�ff�Y��@-p�A��
C��                                     ByP�
  
�          @�=q@�
=���
@p�A�=qC�P�@�
=�fff@,(�A�\)C�
=                                    ByP��  
�          @�G�@�\)��p�?ٙ�AqG�C��R@�\)���\@�A�p�C��                                    ByP�V  �          @��
@˅�#�
?�33Aw�C�
=@˅��  @*�HA�  C�S3                                    ByP��  �          @�@�
=�<��?�33AW�
C�S3@�
=�(�@$z�A�(�C�Ff                                    ByP֢  
�          @���@���L��?��A(�C��H@���(��@�A�p�C��                                    ByP�H  
�          A  @�\�\)?Q�@��C�G�@�\�^{@��As33C���                                    ByP��  
�          A
=@��H�I��@A�A���C��@��H��@{�A�33C�l�                                    ByQ�  
�          A
=@�33��G�@���BG�C�9�@�33?�=q@�(�B�A�                                    ByQ:  "          Aff@�=q��@���B�C�*=@�=q��\)@�{B�C��{                                    ByQ�  �          AQ�@���N{@8��A��C���@����\@tz�A�ffC���                                    ByQ.�  "          A�R@��R�n�R@�An�\C��@��R�.�R@Z�HA��C�B�                                    ByQ=,  �          A	G�@�(��q�@   AYp�C�q�@�(��7�@J�HA��RC�b�                                    ByQK�  	�          A�\@�
=�z=q@:�HA�
=C��@�
=�,(�@��\A�{C�g�                                    ByQZx  
�          A
=@�z��W�@��A�=qC�]q@�z��G�@��B
{C��                                    ByQi  �          A\)@�Q��x��@j�HAɮC��@�Q��=q@���B�C�/\                                    ByQw�  T          A��@أ��w�@���A�C�!H@أ��	��@�ffB
=C�4{                                    ByQ�j  �          A\)@ָR��Q�@�(�A��
C���@ָR�(�@���BQ�C��                                    ByQ�  
�          Az�@Ӆ���H@�33A�ffC�]q@Ӆ�'
=@��B{C�8R                                    ByQ��  �          A  @���xQ�@�G�BC��@������@��B$ffC���                                    ByQ�\  �          A��@���C33@��B(�C�}q@���Tz�@׮B@Q�C�
=                                    ByQ�  �          A\)@����(�@��B?=qC��q@���>k�@���BL=q@�                                    ByQϨ            A33@�����p�@�{BM�\C���@���?��\@��BPA'
=                                    ByQ�N  �          A�H@�33��@�\BS�
C��@�33?�\)@�(�BV  A=G�                                    ByQ��  �          AG�@�p��ٙ�@�Q�BT��C�� @�p�?Tz�@�p�B[{AG�                                    ByQ��  �          A��@��׿ٙ�@�=qBXp�C��q@���?W
=@�
=B_  Az�                                    ByR
@  T          A  @�\)��  @�(�B\�RC��)@�\)?��@�(�B\\)Ahz�                                    ByR�  �          A�@���{@�
=B^�C�E@�?��H@�  B_�RA\                                      ByR'�  �          A��@����
=@��HBn�C�AH@��@�@陚B`��A��
                                    ByR62  �          A
=@��\��  @��Be33C���@��\?�p�@�
=Ba(�A���                                    ByRD�  �          A	�@������@�p�Ba�C��R@��?�  @��BaG�Ar�R                                    ByRS~  �          A�
@�33��p�@�p�Bg=qC�)@�33?�(�@�Bg\)Az{                                    ByRb$  �          A
ff@�
=��@��B`  C���@�
=?!G�@�Bj{A ��                                    ByRp�  �          A\)@�33��ff@���B`�C�Ǯ@�33?�R@�\)Bj�RA�                                    ByRp  �          A=q@��
�Ǯ@�z�BiQ�C���@��
?aG�@�Q�BoffAA��                                    ByR�  �          A��@�����{@�(�BjffC�+�@���?Tz�@�Q�Bq�A8��                                    ByR��  �          A\)@�p���z�@�33Bd�C���@�p�?�@��HBp@���                                    ByR�b  �          A
=@���@�\Bc�RC��f@�?   @�=qBp33@��                                    ByR�  �          A��@�=q���H@�=qB\C��@�=q>\@��HBj=q@�
=                                    ByRȮ  �          A�
@��Ϳ��H@�BX�C�  @���>���@޸RBf�@��                                    ByR�T  �          A�R@�p�����@���B`��C�+�@�p�>��H@�Q�Blff@���                                    ByR��  �          Aff@��
�˅@�33Bd��C�n@��
?8Q�@߮BlffA�R                                    ByR��  �          A ��@�{�@��
BQ�C�n@�{=���@�ffBa��?�                                      BySF  �          Ap�@u��ff@�  Bo�
C���@u?�ff@�G�Br33As�
                                    ByS�  �          @���@�Q�Ǯ@�(�Bd�C�Y�@�Q�?+�@���BlG�A33                                    ByS �  �          @�@z=q��\)@�Bf�C��q@z=q?�R@�33Bo��A��                                    ByS/8  �          @�z�@l(���z�@�G�Bk\)C���@l(�?G�@�z�Bq��A=                                    ByS=�  �          @��H@`  ��\)@�(�Bu=qC�K�@`  ?h��@޸RBy��Ai��                                    BySL�  �          @��@[���@ڏ\Bx=qC��3@[�?��@�33By�A��\                                    ByS[*  �          @�=q@P  ����@��Bv33C�)@P  ?&ff@��B��A4��                                    BySi�  T          @�Q�@W
=��{@׮Bqp�C��=@W
=>���@�\)B�L�@�G�                                    BySxv  �          @�  @N{��@���Br
=C�U�@N{>�p�@�Q�B���@��H                                    ByS�  �          @��@XQ��p�@ϮBn��C�t{@XQ�>�(�@�ffB|G�@�G�                                    ByS��  �          @�
=@[���Q�@ָRBq=qC��@[�?�@���B}{A                                    ByS�h  �          @��@G��@أ�Bs�C�"�@G�>.{@�\B�W
@J=q                                    ByS�  �          @���@>{��
@�z�By=qC���@>{>k�@�{B��@�
=                                    ByS��  �          @�@>{�@�Bu�
C�t{@>{>�@�  B��@#33                                    ByS�Z  �          @�\)@C33���@�=qBuQ�C�%@C33>�\)@ڏ\B���@�G�                                    ByS�   �          @�p�@L�Ϳ�33@�\)Bs�C�S3@L��>��@�p�B�B�A{                                    ByS��  T          @�@E���@�
=BrC��@E�>W
=@�  B�Ǯ@z=q                                    ByS�L  �          @陚@L�Ϳ���@ʏ\Bq�HC���@L��>�@�Q�B~=qA�
                                    ByT
�  �          @陚@ �׿�\)@�
=B��C���@ ��>���@�
=B�\)A��                                    ByT�  �          @�=q?��
�	��@�Q�B��C��R?��
=�\)@��HB��q@33                                    ByT(>  �          @���?����@ָRB�k�C�T{?�=u@�G�B�?�Q�                                    ByT6�  �          @�{@	���   @��B�C��H@	��>\)@ۅB�#�@n{                                    ByTE�  �          @�
=@"�\�@�B{�C�H�@"�\<#�
@�Q�B���>��R                                    ByTT0  �          @��
@(Q���@��HB{�RC�/\@(Q�>.{@��
B�z�@o\)                                    ByTb�  �          @�@&ff���H@ə�Bz��C�|)@&ff=��
@�33B���?�\)                                    ByTq|  �          @��@%���33@ȣ�B{�C��@%�>�@љ�B�
=@;�                                    ByT�"  �          @�  @���@�
=B|�C��\@�����@�=qB��C���                                    ByT��  �          @ᙚ@\)�ff@ȣ�Bz�C�Ǯ@\)���R@�{B��3C�                                    ByT�n  �          @�Q�@�-p�@��
Bs(�C��)@�333@���B�L�C���                                    ByT�  �          @ᙚ@{�p�@�(�Bq�\C��\@{��@��HB�B�C��\                                    ByT��  �          @�@.�R�(�@�p�BqC���@.�R�L��@љ�B���C��q                                    ByT�`  �          @��
@1녿��@���B�8RC��\@1�?!G�@У�B��
AK33                                    ByT�  T          @��@)����ff@�
=B�8RC��
@)��>��H@�(�B�A(Q�                                    ByT�  �          @���@!G���@��HBz�HC�AH@!G��u@�p�B��\C�]q                                    ByT�R  �          @�\)@,���s�
@���BL
=C��=@,�Ϳ��@��B~�C��                                    ByU�  �          @�z�@"�\�{�@���BNG�C�h�@"�\��(�@��B��=C�q                                    ByU�  �          @�z�@p��HQ�@ə�Bj�RC���@p���=q@�ffB���C��
                                    ByU!D  �          @�G�@=q�8��@ǮBk�
C��@=q�^�R@�=qB��)C��                                    ByU/�  �          @�Q�@�
�C�
@��Bhp�C���@�
����@�G�B��C���                                    ByU>�  �          @�ff@*�H�A�@��RB_�\C��3@*�H����@��HB��C��f                                    ByUM6  �          @��@<���)��@�=qBcz�C��@<�Ϳ5@��HB��{C�1�                                    ByU[�  �          @�p�@N{�   @�{Bb�\C��@N{�
=q@��B�C�9�                                    ByUj�  �          @�
=@Vff��R@�ffB`Q�C�@Vff��@��B|  C���                                    ByUy(  �          @�{@^�R�z�@���B_{C�&f@^�R�Ǯ@��Bw��C���                                    ByU��  �          @�R@g���@�z�B]��C�u�@g���=q@У�Bsp�C��                                    ByU�t  �          @�ff@w
=��@�=qBZz�C��f@w
=�#�
@˅Bk�C��{                                    ByU�  �          @�{@e�z�@��B`(�C��@e�#�
@ϮBt{C��
                                    ByU��  �          @�(�@Q��\)@��
B`33C�g�@Q녿z�@ҏ\B|��C�                                      ByU�f  �          @陚@XQ����@��B`�RC��@XQ�\@θRBy
=C�˅                                    ByU�  �          @���@N{�z�@��HBc��C�q@N{��G�@�Q�B}C�'�                                    ByU߲  �          @߮@?\)�
=@��
Bd  C��q@?\)�
=q@ə�B�z�C��q                                    ByU�X  �          @���@!G��)��@�{Be�C�Ǯ@!G��aG�@�
=B��=C�k�                                    ByU��  �          @�33@   �<��@�ffBc�HC�#�@   ���@��B�C�1�                                    ByV�  �          @�p�@&ff�<��@�\)Bb�RC��\@&ff���@ҏ\B���C��                                    ByVJ  �          @�\@ ���@��@�z�Baz�C��@ �׿�@�Q�B��HC�w
                                    ByV(�  �          @���@.�R�3�
@��\B`=qC��@.�R��G�@�z�B�C��{                                    ByV7�  �          @�z�@8���(��@��RBb�RC��@8�ÿQ�@�
=B�33C�                                      ByVF<  �          @�\)@h����
@��BY��C�C�@h�þ�z�@�ffBm��C��)                                    ByVT�  �          @�
=@W�����@��Be  C��)@W���@˅Bx33C��                                    ByVc�  �          @�  @C�
��\@���Bh�C��
@C�
��G�@љ�B�u�C��                                    ByVr.  �          @�Q�@;��#33@�(�Bfz�C�s3@;��333@�33B���C�G�                                    ByV��  �          @�G�@,(��*�H@�
=Bj�C��@,(��G�@�\)B��qC��H                                    ByV�z  �          @陚@
�H�?\)@�G�BnG�C��@
�H��=q@�(�B�#�C�Ф                                    ByV�   �          @�{@4z��  @ƸRBo�C��\@4z����@�33B�W
C��
                                    ByV��  �          @�\)@$z��0��@�p�BjG�C�� @$z�fff@�ffB��RC�T{                                    ByV�l  �          @�z�@#33�333@���Bh  C�(�@#33�z�H@��HB�\C�~�                                    ByV�  T          @��
@-p��0  @�\)Bd�\C�Q�@-p��u@�Q�B�Q�C�J=                                    ByVظ  �          @���@33�<(�@�Bg(�C��@33��z�@У�B���C���                                    ByV�^  D          @�33@  �<��@�G�Bi��C���@  ���@�(�B�B�C��
                                    ByV�  
�          @���@�Mp�@�{Ba  C��@��@�33B��C�\)                                    ByW�  "          @��@p��U�@�B`�C�˅@p���ff@��
B���C���                                    ByWP  �          @�\@��L(�@��RBe�RC���@���z�@ӅB�\)C��3                                    ByW!�  �          @��
@�C33@���BeffC�.@���@�z�B���C��\                                    ByW0�  �          @ڏ\?�
=�N�R@�{Bb33C�o\?�
=���
@˅B��C���                                    ByW?B  �          @�z�?����S33@��Bi�C���?����Ǯ@љ�B�  C�g�                                    ByWM�  T          @�{?�33�g�@�Q�B`
=C��)?�33��33@���B�=qC�@                                     ByW\�  �          @��H?�  �\(�@���Bf��C�?�  ��(�@�  B��3C�                                    ByWk4  "          @�Q�?��R�S33@��HBbQ�C���?��R��=q@У�B�ffC���                                    ByWy�  �          @�=q@�R�N�R@��B`��C�P�@�R��G�@�Q�B�C��                                    ByW��  �          @�z�@Q��fff@���BX�C�G�@Q���@�G�B���C�'�                                    ByW�&  �          @��
?����e@�{B`33C�P�?��Ϳ�z�@�ffB�#�C��                                     ByW��  
�          @�
=?�{�`��@��Bhp�C�˅?�{���
@�z�B��C�H                                    ByW�r  
�          @�=q?��H�\(�@��BhG�C�y�?��H�ٙ�@�{B�\C�O\                                    ByW�  
Z          @���?���|(�@���B]�C�O\?���\)@�(�B�k�C�u�                                    ByWѾ  
�          @��H?W
=�s�
@��
Ba�C�4{?W
=�@�p�B���C��                                    ByW�d  
�          @�p�?��?\)@�ffBpC�^�?���(�@أ�B��
C��                                    ByW�
  �          @�
=?��k�@��HBf�C��\?���@�33B��3C���                                    ByW��  �          @�{�����AG�@���B�G�C�5þ������H@�
=B��qC�#�                                    ByXV  T          @�\��
=�~�R@���B[�\C}���
=�G�@׮B�p�Cr��                                    ByX�  "          @���9���l(�@�Q�BF�Cg���9���ff@�G�Bq�CW޸                                    ByX)�  �          @�R� ���xQ�@��RBRffCr�\� ����R@���B��\Cc��                                    ByX8H  
Z          @���?n{���H@�G�BXQ�C�e?n{��H@��B�ffC�y�                                    ByXF�  "          @�Q�?\����@��BV��C�U�?\�@�ffB�ǮC�q�                                    ByXU�  
�          @�=q?�p�����@�z�BW(�C�
=?�p���@׮B���C���                                    ByXd:  �          @�33?�  ���\@�p�BV�
C��?�  ���@���B��qC��)                                    ByXr�  
�          @�(�?���|��@��HB^�HC�\?����R@��B��C��                                    ByX��  
�          @��
?�  ����@���B\p�C���?�  �z�@�z�B�\C�#�                                    ByX�,  T          @��
?��\���@��B]\)C��?��\�
=@���B�#�C���                                    ByX��  
(          @�(�?��
��=q@��RBW��C�Q�?��
���@��B��C�N                                    ByX�x  "          @�33@?\)�h��@��
BC  C���@?\)���@�(�Bl�C�9�                                    ByX�  
�          @���@�(��\��@���B)G�C�
@�(���@��
BJ(�C��=                                    ByX��  �          @�=q@�Q��h��@�p�B(��C�޸@�Q��G�@�{BK��C�B�                                    ByX�j  
�          @�@��
�p��@�BG�C���@��
�(�@�\)B>Q�C�j=                                    ByX�  
�          @�R@���mp�@��HB!�C��@���
=@��
BCz�C���                                    ByX��  
(          @�ff@�  �k�@�z�B#��C��@�  �z�@��BE�RC��=                                    ByY\  �          @�@�G��a�@�ffB&�
C�AH@�G��
�H@�BG�C��{                                    ByY  	`          @�ff@���_\)@��
B#z�C���@���	��@��HBB�
C���                                    ByY"�  "          @�ff@��e@��HB(�C���@���@��HB@��C�t{                                    ByY1N  
�          @�
=@y���W�@�G�B1ffC��
@y���   @�
=BR�C�h�                                    ByY?�  	�          @�p�@����Q�@�p�B-C�p�@��ÿ���@��\BM{C��                                    ByYN�  
�          @��
@�{�S33@��RB%�C���@�{��@�(�BD�C�R                                    ByY]@  �          @�p�@s�
�g�@�z�B+��C�@ @s�
�33@�(�BN�C�z�                                    ByYk�  "          @�@[��w
=@��B1C��@[��\)@��BX�C�H                                    ByYz�  "          @�@p  �Z�H@�p�B6{C��
@p  ��\@�33BWz�C��q                                    ByY�2  
�          @��
@o\)�>�R@���B?G�C��3@o\)����@�33B\��C�o\                                    ByY��  "          @߮@u�0  @�p�B>ffC�1�@u��33@�BY33C���                                    ByY�~  �          @߮@{��B�\@���B2Q�C�#�@{���  @��BO��C���                                    ByY�$  
(          @�@��
�"�\@�  B!�C���@��
��\)@��B7��C��                                     ByY��  T          @���@���B�\@�G�Bz�C�p�@����z�@���B0��C���                                    ByY�p  
�          @��
@���N{@��RBffC��@���
=@��
B,��C��                                    ByY�  
Z          @��
@�33�<(�@fffA�  C���@�33���R@�ffBffC��=                                    ByY�  "          @�(�@��H�0  @A�A�33C�e@��H��Q�@fffA�RC��\                                    ByY�b  �          @�(�@��\�N�R@\��A�C�j=@��\��
@��
BQ�C�N                                    ByZ  T          @���@����Y��@{�B�C��q@����
=@�(�B!�C�                                      ByZ�  
�          @�z�@�ff�W�@x��B��C��@�ff�@��\Bz�C�T{                                    ByZ*T  "          @�(�@��J�H@��B
�HC�� @���@�ffB$��C�aH                                    ByZ8�  "          @���@�=q�:=q@r�\B   C��@�=q��Q�@��
B�RC�H                                    ByZG�  "          @��
@��
�/\)@Y��A�RC�  @��
��{@|��B
=C���                                    ByZVF  "          @�@�\)�'
=@l(�A�=qC�AH@�\)��@�ffB�C���                                    ByZd�  �          @�(�@�
=�p�@vffB�HC��f@�
=���R@�=qB=qC�c�                                    ByZs�  "          @��
@���@�=qBQ�C�\@���@��RB=qC��=                                    ByZ�8  
�          @��@�Q���R@��B �HC�"�@�Q�aG�@�z�B/�C��                                    ByZ��  T          @���@�
=�(�@��B#��C�U�@�
=��ff@���B7�C�Ff                                    ByZ��  
Z          @�@���%@�=qB0\)C�H�@����z�@�G�BG�C��                                    ByZ�*  
�          @���@l(��6ff@�=qB=z�C�#�@l(���{@��HBX�C�4{                                    ByZ��  
�          @�Q�@X���3�
@�33BD�
C�33@X�ÿǮ@��BaffC��                                    ByZ�v  
�          @أ�@B�\�9��@���BNffC�1�@B�\��{@�=qBm�C�\                                    ByZ�  
Z          @�p�@=p��5@��BO��C�{@=p���=q@�  Bn�HC��)                                    ByZ��  T          @���?�����@�33BN��C�#�?��)��@˅B|�C�7
                                    ByZ�h  	�          @�=q?�33��
=@��B6��C���?�33�j=q@�
=Biz�C��                                     By[  
�          @���?�\)���@�=qB&�C��3?�\)���\@�z�BY33C���                                    By[�  
�          @�?�{��G�@�{BN�C���?�{�9��@�Q�B33C���                                    By[#Z  "          @�Q�?�33���
@�Q�BQ��C���?�33�.{@���B�{C���                                    By[2   
�          @��H?޸R��p�@��RBI�C�O\?޸R�5@�Q�Bw��C��{                                    By[@�  �          @�=q@
=�l��@�z�BS��C�ٚ@
=�
=@�=qB}��C��\                                    By[OL  
|          @�G�@!��4z�@��
B_��C��
@!녿�G�@�33B�u�C���                                    By[]�             @�@2�\��
@�ffBW=qC�&f@2�\��z�@��HBr
=C���                                    By[l�  �          @��@!G��=p�@�G�B?(�C�(�@!G�����@��\Bb�\C�                                      By[{>  �          @�ff?���s�
@�Q�B<
=C���?���.�R@��Bhp�C���                                    By[��  �          @��
?\���\@�z�B;G�C�9�?\�>{@�p�Bi��C���                                    By[��  
�          @�p�?���Q�@�=qB5��C�1�?��J=q@�z�Bd�C�\                                    By[�0  �          @ȣ�?L���mp�@�p�BQ\)C�
?L���#33@��B��C��3                                    By[��  �          @�z�    �*�H@��
B�(�C��    ���@��B�#�C�                                    By[�|  
(          @��?�Q����R@��HB�HC��R?�Q���33@�33BN�C�/\                                    By[�"  �          @�
=?��R��ff@�G�B&{C��R?��R����@�G�BUG�C��                                    By[��  �          @��?Ǯ��ff@���B"\)C�U�?Ǯ���H@��BQ(�C�n                                    By[�n  �          @��?�������@���B/��C�\?����h��@�B^ffC�]q                                    By[�  �          @��?���33@�z�B+�C���?��p  @��B[  C���                                    By\�  �          @�p�?�\)���@�G�B/ffC���?�\)�n�R@��RB^�C�^�                                    By\`            @�33?�(���G�@��B/33C�#�?�(��k�@�z�B]��C�'�                                    By\+  T          @�z�@   ����@��RB"�C�` @   �n�R@��BN��C�)                                    By\9�  �          @ۅ@z����@��\B4
=C�]q@z��HQ�@��B\�HC�H�                                    By\HR  �          @ڏ\@ ���z=q@��B<(�C�Q�@ ���2�\@�ffBb��C��)                                    By\V�  �          @�(���  ��z�@`��BffC�uþ�  �s33@���BDG�C�                                      By\e�  �          @��ÿk����@xQ�B�C�Z�k��mp�@�  BL�\C��                                    By\tD  �          @�\)>u�s33@��RBW\)C���>u�)��@�z�B�k�C���                                    By\��  �          @��H�E����@�z�B#33C�g��E��r�\@���BQ�HC�Ff                                    By\��  T          @��H�xQ���  @�ffB)\)C��xQ��c33@���BWG�C~                                    By\�6  �          @�=q��\)����@��HB"�C}:`\)�g�@�{BO�CyB�                                    By\��  �          @�Q�333���H@s33BffC�� �333�~{@�ffBE�C��)                                    By\��  T          @�
=�������\@��B)��C�{�����g
=@���BV��C|��                                    By\�(  �          @ƸR���\���H@�{B+=qC}�����\�Z=q@�\)BWp�Cy��                                    By\��  �          @�
=�(�����@>{A�Q�Cr޸�(��e@qG�B'��Cn�)                                    By\�t  �          @����R��33?�33A��Cq����R��(�@1�A�{Co�                                    By\�  �          @�z��'���?�A�=qCp� �'���Q�@$z�Aٙ�Cn\)                                    By]�  �          @����R�\��  ?�p�An�RCi�{�R�\���@ffA�=qCg\)                                    By]f  �          @�=q��ff����@#�
A؏\C�����ff����@^�RBC�(�                                    By]$  �          @�ff=�������@?\)A���C��\=������@z�HB'C��f                                    By]2�  �          @�Q�����p�@(��A�G�Cv� �����@`��B�RCsp�                                    By]AX  �          @�33�,(���{@p�A�=qCp0��,(����@EA�p�Cm.                                    By]O�  �          @�33�|(��+��}p��;�CVB��|(��5���ff����CW��                                    By]^�  �          @�����{�33��\��33CN
��{�p��������RCRc�                                    By]mJ  �          @�{��(��_\)>�Q�@e�CX����(��Vff?��
A#\)CW�f                                    By]{�  
�          @�(�����l(�?@  @�  CVO\����^�R?�Q�AJ=qCT�                                     By]��  �          @���=q�z�H?��HAn�HCY����=q�aG�@�A�CV��                                    By]�<  �          @�(������ff?�(�A|��C_�f����s33@   A�C\�                                     By]��  �          @�=q��=q�^{>aG�?���CS�H��=q�W
=?aG�@�{CS
                                    By]��  �          @�=q���\�N{�����CTT{���\�P��=�\)?333CT��                                    By]�.  �          @�=q���a녿#�
���HCV@ ���e=#�
>��
CV��                                    By]��  �          @Ӆ�����c�
�#�
��=qCT�=�����g�=#�
>�{CU5�                                    By]�z  �          @У������e��G��s33CU�f�����c33?�@�Q�CUT{                                    By]�   �          @��
�\�����\@��A�Q�CjxR�\����33@@  A�
=Cg��                                    By]��  �          @�  �8����G�?�Q�A�=qCmu��8����33@/\)A�\)Cj                                    By^l  �          @�Q��P  ���\?��A�33Ci!H�P  �y��@)��A�=qCfE                                    By^  
�          @��\�qG�����?�ffA�ffCc
�qG��i��@!G�A��C`�                                    By^+�  �          @�{���H����?��Ap  C`�����H�k�@  A�{C]�R                                    By^:^  �          @�G������j�H?�
=A��RC]� �����S33@z�A���CZ��                                    By^I  �          @�p����H�c�
?�{A���C]����H�L��@�RA�{CZ�                                    By^W�  �          @�G���33�S33?�AH��CZ�{��33�B�\?�  A��CX}q                                    By^fP  �          @~{�333�'
=�n{�Z�\C_  �333�.�R��G���\)C`T{                                    By^t�  �          @��\�e��4zῼ(���33CZB��e��B�\�p���5G�C\Y�                                    By^��  �          @�\)���
���=�G�?�  CP�H���
��?�@���CP
=                                    By^�B  �          @�Q��y�����?��A��CM� �y����G�@�AܸRCI=q                                    By^��  �          @�Q��fff����@=qA�z�CO��fff���@,��B��CI�                                    By^��  �          @����o\)�Q�@�\A�Q�CT���o\)��Q�@*�HB��COz�                                    By^�4  �          @��
>8Q쿧�@@  B�L�C��R>8Q�E�@K�B�=qC���                                    By^��  T          @��>�p���ff?�G�B-C��)>�p���G�?�G�BTC�33                                    By^ۀ  T          @���"�\�|��?�p�A��HCmO\�"�\�c33@(��A�p�Cjk�                                    By^�&  �          @�p��8Q����?�  A���Ck�{�8Q��w�@p�A�{Ci^�                                    By^��  �          @��\�I������@  A�Q�Cg��I���dz�@:=qA���Cd�{                                    By_r  �          @�33�R�\�{�@z�A�  Cf{�R�\�aG�@.{A�Cb�f                                    By_  �          @�z��\)���@B�\B  Cq�\)�e�@mp�B%33Cn�                                    By_$�  �          @�  �L����33?���A��Ci�H�L���~{@$z�AՅCg)                                    By_3d  �          @��J=q����@=qA��
Cg޸�J=q�c33@C�
B  Cdc�                                    By_B
  �          @��:�H�~�R@0  A�33CiǮ�:�H�]p�@X��BG�Ce�)                                    By_P�  �          @����g
=�{�@��A���Cch��g
=�`��@5�A��C`!H                                    By__V  �          @����Y���\(�@#�
A�\)CaT{�Y���=p�@FffB
(�C]�                                    By_m�  �          @��R�L���a�@�  BA��C�` �L���3�
@���Bh\)C��
                                    By_|�  �          @�����vff@G�B  C�)���Q�@mp�B@�HC�p�                                    By_�H  T          @�p�?�
=��=q@(Q�A���C���?�
=�e�@P��B!�C�޸                                    By_��  �          @�z�?���dz�@fffB-\)C�E?���;�@�z�BPz�C�Ff                                    By_��  �          @��
?B�\�W
=@�33BG�C�aH?B�\�(��@��HBl�\C��                                    By_�:  �          @�33?��R�_\)@[�B"�C��?��R�8Q�@}p�BB  C�Z�                                    By_��  �          @�{@�H�w
=@��A�=qC��@�H�Z=q@C33BG�C��
                                    By_Ԇ  �          @��
@A��H��@�RA��HC��@A��,(�@=p�BQ�C�1�                                    By_�,  �          @�G�@W��=p�@�A�p�C�aH@W��!�@7�B	�C���                                    By_��  �          @��R��{��z�@�\A�=qCx� ��{�n{@;�BQ�Cv�
                                    By` x  �          @���?
=����@^�RB!\)C�#�?
=�\(�@��\BE�C��)                                    By`  �          @���@J=q�U�@S33B�\C�� @J=q�1G�@q�B(  C�j=                                    By`�  �          @�  @3�
���\@|(�BffC���@3�
�i��@��\B333C��\                                    By`,j  �          @Ӆ@7���
=@z=qBQ�C�U�@7��s33@��B/�C��f                                    By`;  �          @��?�(���ff@I��A�G�C�J=?�(���z�@x��B {C�E                                    By`I�  T          @�G�?�
=��  @b�\B�RC��R?�
=�x��@�ffB0ffC�.                                    By`X\  �          @�z�@*�H��z�@q�BffC��=@*�H�o\)@�p�B/Q�C���                                    By`g  �          @�@�R��ff@^�RB	�C��\@�R�vff@�(�B(�C�]q                                    By`u�  �          @�z�@)����(�@UBz�C��R@)���s�
@~�RB"=qC�h�                                    By`�N  �          @�  @2�\��33@K�A���C���@2�\���@vffB{C�AH                                    By`��  �          @ȣ�@@  ���@:=qA�p�C�ff@@  ���@eBC���                                    By`��  �          @ʏ\@_\)�hQ�@{�BC��\@_\)�?\)@�{B/�
C��3                                    By`�@  �          @\@K��k�@g
=B\)C�k�@K��E@�(�B+C��=                                    By`��  �          @�{@n{�h��@�B  C��@n{�>{@�B2  C���                                    By`͌  �          @�33@dz����@��\B�C�S3@dz��`  @��B+C��=                                    By`�2  �          @�\)@r�\��33@z�HB�RC��@r�\�n{@���B!C��                                    By`��  �          @�p�@XQ���=q@�
=B33C�@XQ��h��@��B0=qC�t{                                    By`�~  T          @�ff@;����@�=qBC�� @;����\@�\)B-{C��
                                    Bya$  �          @�ff@���(�@o\)B=qC���@�����@��B!
=C�                                    Bya�  �          @���@(���=q@���B
=C�.@(���p�@�Q�B*�C���                                    Bya%p  �          @�
=@.�R���
@�p�B�C��)@.�R���R@��\B/�C�s3                                    Bya4  �          @�
=?�G���z�@qG�B�C��H?�G�����@���B#��C�]q                                    ByaB�  �          @�
=?�  ��G�@mp�B �C���?�  ��ff@�Q�B!��C�u�                                    ByaQb  �          @�{>�ff����@g
=A���C�33>�ff��=q@�p�BffC�p�                                    Bya`  �          @��?��\���@i��B�\C�q�?��\��\)@��B"�\C�*=                                    Byan�  �          @�Q�?xQ���G�@UA�33C��)?xQ���  @�(�B�HC�5�                                    Bya}T  �          @��>8Q���(�@s33B  C��>8Q�����@��B(��C��                                    Bya��  �          @�\)?Q����@z=qB33C�^�?Q���ff@��
B0=qC��                                    Bya��  �          @�33@(Q���G�?�p�A�
=C�c�@(Q����R@(Q�A�33C�AH                                    Bya�F  T          @��
@Q���  @.{A�Q�C��=@Q���=q@X��B�\C�~�                                    Bya��  �          @�=q@   ��(�@k�Bp�C�G�@   �u�@�  B233C��\                                    Byaƒ  �          @���@\)��  @h��B��C��@\)�l��@�{B0\)C��
                                    Bya�8  �          @ȣ�@����  @�ffB(Q�C�l�@���XQ�@��RBB�C���                                    Bya��  �          @�(�@6ff�e�@�G�B%C�C�@6ff�@  @�  B=��C��                                    Bya�  �          @Ǯ@5��x��@}p�B��C�f@5��S�
@�ffB6z�C�B�                                   Byb*  �          @Ӆ�8Q���G�@vffB33C�  �8Q���
=@�G�B/p�C��H                                   Byb�  �          @�Q�
=��{@�\)B33C�� �
=��=q@���B;
=C�O\                                   Bybv  �          @��ÿ������@h��B(�C~{������ff@�33B��C|s3                                    Byb-  �          @���(�����
@P��A�Cr���(����z�@z=qB�CpW
                                    Byb;�  �          @����!���33@J=qA�\Cs���!���(�@s�
Bz�CqY�                                    BybJh  �          @�(��'���  @C�
Aܣ�Cs���'�����@n�RB	\)Cqc�                                    BybY  T          @�(���Q���=q@�(�B.��C��f��Q��|��@�ffBM
=C�n                                    Bybg�  
�          @ə�=��
��
=@�B?�C���=��
�dz�@�{B^�C���                                    BybvZ  �          @�ff>.{��\)@�z�B7��C��>.{�u@�BU�
C�J=                                    Byb�   �          @�>\�~�R@�33B=��C�>\�XQ�@��\B[\)C�>�                                    Byb��  �          @��H<��
�n�R@�
=BH��C�'�<��
�G�@��BfffC�/\                                    Byb�L  �          @��
�\(��z=q@�  B:C��Ϳ\(��U�@�
=BW��C��                                    Byb��  �          @�{��(��~{@���B1��Cy�ÿ�(��Y��@��
BM(�Cv�                                    Byb��  �          @�{�z�H�U@�Q�BWffC}��z�H�,��@���Bs=qCy�q                                    Byb�>  �          @�ff�Y���7�@�BeffC}�ͿY�����@�  B��=Cyff                                    Byb��  �          @�  ?��z�H@��B�W
C���?���z�@�{B�(�C�/\                                    Byb�  �          @�33>u�1�@��Be��C�y�>u�p�@�p�B�.C��                                    Byb�0  �          @�
=?�z��<��@�(�B]  C���?�z��z�@��RBuG�C�˅                                    Byc�  �          @��@�����H@��
BZ�
C�b�@����33@��\Bk=qC��H                                    Byc|  �          @��ý�\)����@�=qBO{C��
��\)�\)@��
Bk�\C���                                    Byc&"  �          @�\)�c�
��33@�=qB<{C�C׿c�
���H@�{BW��C��                                     Byc4�  �          Aff�����H@�33B{C}xR����p�@��B5��C{��                                    BycCn  �          A
=�����@���B�RC}ff����  @���B3z�C{�                                    BycR  �          A�R��Q����@���B �
C�Q쿸Q���(�@��RB;�C�                                    Byc`�  �          A33��  ��(�@��B+�C�H��  ���@�\)BF33C�33                                    Byco`  �          A녿333����@�\)BH�C���333���@ٙ�Bd  C�h�                                    Byc~  �          A���33��  @��B;G�C�3��33���@�33BV{C�(�                                    Byc��  �          AG����R���\@���B:z�C������R��=q@�p�BU{C�
                                    Byc�R  �          A{������@�Q�BA�C��=��������@ۅB\  C�{                                    Byc��  �          A������  @��B6��C�5ÿ�����@�{BQ
=C~u�                                    Byc��  �          A	�����R���
@��HBHffC��f���R���@�Bb�\C~ٚ                                    Byc�D  �          A������R@���B0  Cy���Q�@ǮBH��CvG�                                    Byc��  �          @�z��C33���\@�=qB(�Crk��C33��G�@�
=B�\Cp\                                    Byc�  T          @�33�3�
���H@�G�B�CtG��3�
����@�B��Cr�                                    Byc�6  �          @��\�HQ���\)@~{A��
CrY��HQ���\)@�(�B{CpB�                                    Byd�  �          @�33�a����@O\)A�{Cp5��a���  @z�HA�G�Cnz�                                    Byd�  �          @�z��X���ڏ\?�Q�AdQ�Cs���X����=q@,��A���Cr��                                    Byd(  �          @�(��g���
=@)��A��Cp�q�g���(�@W
=A�=qCoff                                    Byd-�  �          @�
=�N{�У�@N�RA���Cs�R�N{�Å@|(�A�Cr:�                                    Byd<t  �          @����33��{@Z�HAΏ\C{��33��Q�@���A��Cy�
                                    BydK  �          @�(��(���  @P  A¸RCz)�(����H@~�RA�RCx�3                                    BydY�  �          @��(���ۅ@>{A�{Cx��(���Ϯ@n{A߅Cw�{                                    Bydhf  �          @�{�\)��@\)A�z�C|���\)�ۅ@QG�A��HC{�3                                    Bydw  �          @�p��!����H@B�\A��HCy���!��θR@qG�A�  Cx��                                    Byd��  �          @���
=��\)@9��A�\)C}8R�
=��(�@i��A�\)C|^�                                    Byd�X  
�          @���ff����@<��A�Q�C{0��ff��G�@k�A߅Cz:�                                    Byd��  �          @��\��R��(�@.�RA��Cz5���R�љ�@]p�Aҏ\CyG�                                    Byd��  �          @���R�\��{@	��A�z�Cs�\�R�\��p�@7
=A��RCr�)                                    Byd�J  �          @����4z��ə�@tz�A�Cu�)�4z����@�
=B
{CtB�                                    Byd��  �          @��J=q�Ϯ@QG�A��HCt��J=q��33@|(�A��Cr��                                    Bydݖ  �          @�
=�J�H��p�@>{A�
=Ct���J�H��=q@j=qA��HCsY�                                    Byd�<  �          A�c�
��(�@��A�
=Cr�f�c�
���H@FffA��
Cq�H                                    Byd��  �          A ���j=q�޸R?�(�AE�CrG��j=q�׮@(�A��Cq�                                    Bye	�  �          A��o\)���?�
=A"�\Cr��o\)���
@
=qAw
=Cqs3                                    Bye.  �          AQ��p������?��@�
=Cr�3�p����z�?��AH(�Cr8R                                    Bye&�  �          Aff�z=q��\?�(�AG�Cq��z=q���?�(�AX��Cqc�                                    Bye5z  �          A\)���\��p�?�33AO�CpT{���\��@(Q�A��RCo�                                     ByeD   
�          A�R��Q���G�@(��A���Cm���Q��Ϯ@Tz�A�33Cl��                                    ByeR�  �          A���\)����@1�A���Cn���\)��
=@]p�A�
=Cl��                                    Byeal  �          A�H�~{�Ϯ@@  A�=qCn���~{����@h��AӮCm0�                                    Byep  �          A�R�w����
@Tz�A�Cn� �w���Q�@|(�A�
=Cm8R                                    Bye~�  �          A����
���@@  A��Cl޸���
��\)@g�A�  Ckn                                    Bye�^  �          Ap����\��=q@>�RA��HCm����\���@fffA��Ck��                                    Bye�  �          A����
��  @C33A���Cn�)���
���@l��A�p�CmL�                                    Bye��  �          AQ���  ��p�@#�
A�  Cnk���  ��(�@N�RA�z�CmQ�                                    Bye�P  �          AQ�������p�@(�An=qCp��������p�@8Q�A�ffCo�{                                    Bye��  �          A�R�g���  @
�HAv�RCr��g���Q�@6ffA��Cq�
                                    Bye֜  �          A������љ�@'�A���Cnz�������Q�@P  A��\CmT{                                    Bye�B  �          A���
=��ff@HQ�A�z�Ci���
=���
@l(�AٮCg��                                    Bye��  T          A=q��(���=q@L��A�\)Cj(���(���\)@qG�A��HCh�
                                    Byf�  �          A����z���Q�@R�\A�  Cg)��z����@u�A��Cec�                                    Byf4  �          A ��������@G
=A�Q�Cb�H������H@g
=A��
Ca�                                    Byf�  �          A�������=q@Mp�A��HCd�{������@n{A��Cb�
                                    Byf.�  �          A��z����
@N�RA�G�Cd����z���G�@o\)Aۙ�CcB�                                    Byf=&  �          A z���=q��ff@[�A�Q�Ca��=q���@z=qA�RC_�=                                    ByfK�  �          A����
���@XQ�Aģ�Cb����
��
=@w
=A�
=C`#�                                    ByfZr  �          A����
���@QG�A�G�Cb\���
��\)@p  Aݙ�C`:�                                    Byfi  �          A��������@J�HA��HCa&f�����ff@h��A֏\C_aH                                    Byfw�  �          A Q�������p�@P��A��Cf���������H@qG�A�  Cd�3                                    Byf�d  �          A ���u��љ�@&ffA�
=Co��u���G�@L(�A�33Cn�H                                    Byf�
  �          A Q���G�����@!G�A��RCk����G�����@EA��Cj��                                    Byf��  �          AG�����ƸR@,��A��
Cj�H�����{@P  A���Cik�                                    Byf�V  
�          A ����G����@%A�  CkǮ��G�����@I��A�{Cj�f                                    Byf��  �          @�
=��=q��(�@,(�A��
Cjٚ��=q���@N�RA�\)Ci��                                    ByfϢ  �          @�����H���@3�
A�=qCj\���H���R@U�A�33Ch�                                    Byf�H  �          @��R��ff��z�@5�A���Ck�H��ff���@W�A�Q�Cjc�                                    Byf��  �          @�\)�}p����
@,(�A��
Cn#��}p��Å@O\)A�(�Cm�                                    Byf��  �          A   �*=q��R?��@��
CzW
�*=q��\?�  AJ{Cz�                                    Byg
:  �          A ���)����R?��
A�\Czk��)����=q?�A^=qCz{                                    Byg�  �          A=q�xQ���Q�@-p�A�G�Co8R�xQ���  @P��A��Cn.                                    Byg'�  �          A��r�\��ff@
=A��\Cp}q�r�\��
=@<(�A��Co��                                    Byg6,  �          @�\)�[����H@�RA�Cs\)�[���(�@3�
A��Cr��                                    BygD�  T          @�{�X����G�@ffA�Cs}q�X�����@;�A���Cr�3                                    BygSx  �          @��R�\(���G�@z�A�p�Cs#��\(����@8��A�ffCr\)                                    Bygb  �          @�
=�W
=�ٙ�@�A�{Cs���W
=��=q@@  A�
=Cr�                                    Bygp�  �          A z���ff���
@[�AЏ\Ci@ ��ff���@y��A�p�Cg�                                    Bygj  �          A{��ff����@��RB�CX}q��ff�h��@�G�B�CU�q                                    Byg�  �          Ap���  ����@���A��C_���  ��ff@��RBp�C]�{                                    Byg��  �          A ������
=@z=qA���C[z�����(�@���B Q�CYB�                                    Byg�\  �          A�H������
=@��\A�p�C]��������@��RB�C[u�                                    Byg�  �          Az�������z�@��
A�z�Ca�H��������@���B��C_�\                                    BygȨ  �          Az���{���@�Q�A�G�Cc��{���R@�BG�Ca{                                    Byg�N  �          @�
=�����  @Z=qAˮCf������ff@vffA�(�Cds3                                    Byg��  �          @�=q��  ����@vffA�z�Ch(���  ��=q@�Q�B�
CfL�                                    Byg��  �          @�=q��33��{@R�\A�ffCi���33����@n{A��
Cg�                                    Byh@  �          @�  �l(����
@4z�A�p�Cm�)�l(����
@R�\AУ�Cl�q                                    Byh�  �          @��R�Z�H�љ�@p�A�p�Crp��Z�H��33@.�RA�{Cq�3                                    Byh �  T          @�33�B�\��=q?˅A;\)Cv�q�B�\��p�@
=qA~�RCvJ=                                    Byh/2  T          A��W���33?�ffANffCt�)�W���{@
=A�(�Ct{                                    Byh=�  �          Aff�Z�H��  ?�{A��Ct�R�Z�H���
?�
=A[\)CtQ�                                    ByhL~  �          A�
�QG�����@�\Ae�Cu�
�QG���33@'
=A���CuJ=                                    Byh[$  �          A���I�����H@  A{\)CvǮ�I����z�@4z�A���Cv33                                    Byhi�  �          A�
�Fff����@��A�Cv���Fff��\@5�A��RCvT{                                    Byhxp  �          AQ��U���
@!�A��\Ct�
�U����@Dz�A��HCt#�                                    Byh�  �          A��N{����@c�
A��HCt���N{��\)@�=qA��HCs�)                                    Byh��  �          AQ��J=q���H@X��A��
Cu0��J=q�љ�@z=qA��
Ct@                                     Byh�b  �          A��AG����
@g�A�  CvJ=�AG���=q@�(�A�(�CuW
                                    Byh�  �          A�9����
=@`  A�Q�Cv��9����@�  A�ffCu                                    Byh��  �          Ap��4z���33@L��A���Cw�H�4z����H@mp�A��HCv�{                                    Byh�T  �          A Q��=p���z�@W
=A�Q�Cu�3�=p����
@vffA�  Cu
=                                    Byh��  T          @��R�<(���z�@N�RA��Cv+��<(���(�@n{A�G�CuL�                                    Byh��  �          @����G��ۅ@�A�33Cu���G���p�@5A��\Ct�                                    Byh�F  �          @�ff�O\)��(�@	��A�Cs��O\)��ff@(��A��\CsO\                                    Byi
�  �          @����Fff�Ӆ@)��A���Ct���Fff����@H��A�z�Ct.                                    Byi�  �          @���,(����@p  A��HCw=q�,(��Å@�
=B{CvB�                                    Byi(8  �          A ����H��ff@q�A߅Cz&f��H����@�Q�A�\)CyL�                                    Byi6�  �          A ���%���\)@���A��
Cx@ �%����@��B�\Cw@                                     ByiE�  �          A   �;���p�@�z�A��
Ct���;���33@��\B  Csp�                                    ByiT*  �          @�z��!���@j�HA�=qCw� �!�����@��B�\Cv�=                                    Byib�  �          @���{��ff@i��A�33C{  �{��p�@��B Q�Cz0�                                    Byiqv  �          @�\)�xQ���33@�G�B33Ch� �xQ�����@���B=qCf�                                    Byi�  �          @��\��ff���R@���B�
Cb�H��ff��(�@�\)B��C`��                                    Byi��  �          @���������
@�z�B33Cc
=������G�@��RB
=C`�                                    Byi�h  �          @�\)��������@�z�B\)C`�3�����s33@�B$�C^z�                                    Byi�  �          @�
=�0  ��{@�{B33Cr��0  ���@�G�B"�Cp}q                                    Byi��  �          @�z��a���G�@��B�HCh
�a���ff@�p�B'�Ce�                                    Byi�Z  �          @�ff�0�����R@��
B��Cr
�0�����@�\)B  Cp��                                    Byi�   �          @�
=�(����H@\)Bp�Cx�H�(�����@��
BffCw��                                    Byi�  �          @�
=�����G�@��B
Cxٚ������@��B�Cw�                                     Byi�L  �          @����\)���
@�=qBp�Cr��\)��G�@���B-��CqB�                                    Byj�  �          @�33��R��(�@�(�B.�CtG���R����@�{B<ffCrxR                                    Byj�  �          @���z�����@�z�Bz�Cu�q�z����\@�\)B%�HCtY�                                    Byj!>  �          @��=p����
@�z�B�Cp��=p���=q@�\)B�Cn��                                    Byj/�  �          @��ÿ�(��tz�@���BQ{Cr���(��[�@���B^�Cp.                                    Byj>�  
�          @�(���
����@�  B2
=Cs��
��@���B?�Cq
                                    ByjM0  �          @�=q�!G����\@��B<�CnQ��!G��mp�@�BI�Ck��                                    Byj[�  �          @��H�"�\��
=@�Q�B6{Cn�3�"�\�w�@�G�BCG�Cl��                                    Byjj|  �          @�{�Y���u�@�p�B)�Cdff�Y���`  @�p�B4�Ca�)                                    Byjy"  �          @��H�J�H���@�33B"��Ci���J�H�\)@�(�B/{Cg�{                                    Byj��  �          @��
�X����\)@���B �Cjc��X����z�@��RB,�HChW
                                    Byj�n  �          @�\�C33��  @�G�B0�
Ck�
�C33��(�@��\B=\)Ci�\                                    Byj�  �          @��7
=���R@�
=B<�Ck�\�7
=�u@��BH��CiL�                                    Byj��  �          @�(��������@�ffBJ\)CoO\����h��@�ffBWG�Cl��                                    Byj�`  �          @�\�
=q�x��@��BRG�Cp���
=q�_\)@��B_ffCnJ=                                    Byj�  �          @��Ϳ����x��@\B]�Cw�)�����^�R@�=qBk  CuE                                    Byj߬  �          @�33��33�c�
@ÅBbQ�Cq��33�I��@ʏ\Boz�Cn�                                    Byj�R  �          @��H�?\)��G�@��B1Q�Ck(��?\)�|��@�(�B=z�Ch�H                                    Byj��  �          @�(���R����@���BHp�CnT{��R�h��@���BU
=Ck�q                                    Byk�  �          @����y��@�  BQffCqp����`��@��B^ffCn�H                                    BykD  �          @�=q����g
=@�Q�BZ��Crc׿���N�R@��Bg�HCo��                                    Byk(�  �          @�(���  �^{@�z�BX�
Cs&f��  �G
=@�33Be��Cp��                                    Byk7�  �          @�(���ff�B�\@��Bg\)Coff��ff�*�H@���Bs�HCl\                                    BykF6  �          @ۅ��  �Q�@���Be�Cq���  �9��@�\)Bq��Cn�f                                    BykT�  �          @�z��ff�\(�@�ffB_
=CraH��ff�Dz�@��Bk�HCo�
                                    Bykc�  �          @�
=���r�\@���BQ�
Cs&f���[�@�Q�B^��Cp                                    Bykr(  �          @�{��Q��j�H@�=qBUG�Cr+���Q��S33@�G�Bb
=Co�H                                    Byk��  �          @���z��W�@�p�B\ffCnaH�z��@  @��
Bh��CkY�                                    Byk�t  �          @�
=���R�p��@���BQCr����R�Y��@�  B^z�Co��                                    Byk�  �          @��� ���?\)@���Bl��Cl)� ���&ff@�ffBxffChW
                                    Byk��  �          Ap���33��  @J=qA�\)C��=��33����@g
=AծC�aH                                    Byk�f  �          AG������=q?���A�\C�@ ����� ��?�=qAK�
C�4{                                    Byk�  �          A ���z��˅@�33B�C{�R�z��\@��BC{.                                    Bykز  �          @�p���H��G�@��RB@  Cq���H��p�@�\)BL�Co�f                                    Byk�X  �          @�\��R�s33@�Q�BRz�Co�H��R�[�@��B^�Cm�                                    Byk��  �          @�Q��z��J�H@�BpffCo���z��1G�@��
B|ffCkp�                                    Byl�  �          @��(��
=@�
=B��
C_��(���@�\B�z�CY^�                                    BylJ  �          @�33��
�{@�\B���Cb���
��@�
=B��fC]E                                    Byl!�  �          @����  �(��@�
=B~(�Ce���  ���@��
B�B�C`c�                                    Byl0�  �          A33�,(��:�H@�z�Bs��Ccff�,(���R@��B}��C^��                                    Byl?<  �          A��G����@�33BP�Ch���G��l(�@��HB[�Ce�{                                    BylM�  �          A(��(Q���
=@�ffBL�
Co���(Q���=q@�
=BX��Cm{                                    Byl\�  E          A ���$z����R@�p�BH�RCp{�$z����\@�{BT�RCm�=                                    Bylk.  �          A���+���z�@�B-\)Cs���+�����@�  B9��Cq��                                    Byly�  �          A��>�R���R@QG�A�=qCys3�>�R���@o\)A�Q�Cx�                                    Byl�z  �          AG��@  �Q�@�Aj�RC{��@  �	��@<��A�Cz                                    Byl�   �          A���E��
�\@#�
Ax��Czk��E���
@E�A���Cz�                                    Byl��  �          A  �<(��33@�HAlz�C{Y��<(����@<��A��\C{                                    Byl�l  �          A��4z��(�@G�A]��C|(��4z��	@2�\A�33C{�)                                    Byl�  �          AQ��=p��Q�@  AZ�HC{\)�=p��	@1�A���C{�                                    BylѸ  �          A���QG��	@#33Aw�
Cy8R�QG��
=@C�
A��Cx�
                                    Byl�^  �          A���O\)�
{@(�Al��Cyk��O\)��@<��A�=qCy\                                    Byl�  �          A���Vff�	p�@{ApQ�Cx���Vff��H@?\)A��
CxW
                                    Byl��  �          A���Dz���@z�Aa�Cz�
�Dz����@5�A�z�CzE                                    BymP  �          A���QG��
=@
�HAR=qCyff�QG����@+�A��RCy{                                    Bym�  �          AQ��L(���@ffAK�Cy�H�L(��	G�@'
=A~�HCy�{                                    Bym)�  �          A
=�N�R�	@
=AO
=Cys3�N�R��@'�A��Cy#�                                    Bym8B  �          A�\�>�R�\)?�(�A@��C{)�>�R�	�@�RAtz�Cz�
                                    BymF�  �          A�\�G
=�33?޸RA+
=Cz\)�G
=�	G�@��A^ffCz)                                    BymU�  �          Aff�@  ��?���A2=qC{��@  �	��@�AeCz                                    Bymd4  �          A{�:=q�33?���A@  C{� �:=q�	�@p�As�C{=q                                    Bymr�  �          A{�>{�
=?�A<��C{)�>{���@�Ap(�Cz�
                                    Bym��  �          A��S33�Q�@ ��AF{Cx���S33�=q@ ��Ax��Cx��                                    Bym�&  �          A���I���	?�A5�Cy��I����
@Ah  Cy�f                                    Bym��  �          Ap��>�R�
=q?�
=A?\)Cz�q�>�R�(�@(�ArffCz�R                                    Bym�r  �          AG��@  ���@��AZ{Cz���@  ��\@,��A��\CzaH                                    Bym�  �          A���E���
@p�A\Q�Cz
=�E����@-p�A��Cy�R                                    Bymʾ  T          A�H�Fff�  @ ��A~=qCyff�Fff�p�@?\)A�Q�Cy                                    Bym�d  
�          A��3�
�?c�
@���C{p��3�
���?���A�C{O\                                    Bym�
  �          A���6ff�z�?\(�@�z�Cz�q�6ff��?���A�Czٚ                                    Bym��  �          A���&ff�G�?�(�@�
=C|���&ff��
?��HA2�RC|}q                                    BynV  �          A���R�G�?�(�A{C}k���R��?��HAM��C}8R                                    Byn�  �          A���H�{?�  A{C}޸��H���?޸RA5C}��                                    Byn"�  �          Ap�����?�z�@���C~�f���{?�33A,Q�C~                                    Byn1H  �          A=q�
=q�	p�?E�@�
=C�)�
=q�z�?��\A�C                                    Byn?�  �          A
=��Q���ff�������C�(���Q���ff>L��?�
=C�(�                                    BynN�  �          A(���{� �׽��O\)C�xR��{� z�>�33@p�C�w
                                    Byn]:  �          A�\����p��!G����
C�n����녾#�
��{C�q�                                    Bynk�  �          A�ÿ�\)�{���H�W
=C�` ��\)�ff�#�
���
C�b�                                    Bynz�  �          A  ������W
=��(�C��������>��?�C���                                    Byn�,  T          A���G�� (�?�@eC~(��G���
=?}p�@ٙ�C~{                                    Byn��  
�          @���"�\��R?�ffA=qC{(��"�\���
?޸RAH��Cz�3                                    Byn�x  �          @�ff�(�����
?��A�\Cy���(������?��HAL��Cyh�                                    Byn�  �          @���{��=q?�G�A�Cz��{�׮?�z�AP(�Cy�)                                    Byn��  �          @�\)�
=q��G�?��A\)C|Y��
=q�ָR?���A:ffC|+�                                    Byn�j  �          @����G���33?�A4  C}�{�G���Q�?���Ag33C}\)                                    Byn�  �          @��
����  ?
=@�G�C|�{���ָR?}p�@�\)C|xR                                    Byn�  �          @�
=��(���?c�
@�33C
=��(����
?�G�A-G�C~�f                                    Byn�\  �          @أ׿\��(�?��AS33C�LͿ\����?�z�A��C�1�                                    Byo  �          @�  � ���Ӆ?���A��C})� ����G�?�p�AC�
C|�                                    Byo�  �          @����(���\)@   A���C{Q��(���33@Q�A��Cz�R                                    Byo*N  �          @�33����
=?��A�ffC{z�����33@  A�C{#�                                    Byo8�  �          @�(��G����@!�A�
=C{��G�����@8Q�A�ffC{\                                    ByoG�  �          @�\)�fff�!�@�ffB�k�Czp��fff��@��HB��Cw��                                    ByoV@  
�          @ə���\)�(�@���B�(�C��R��\)�@�B���C�0�                                    Byod�  �          @�
=�B�\�z�@��B�u�C��H�B�\����@�B�#�C�.                                    Byos�  �          @�p����
=q@�z�B��3C��R�����@��B�aHC��q                                    Byo�2  �          @θR�J=q�Q�@���B���Cy�R�J=q���
@�z�B��HCv�                                    Byo��  �          @��
�����@�(�B�ǮC��{����z�@�Q�B�L�C                                      Byo�~  �          @��Ϳ���-p�@���B�z�C��=����
=@�B�C                                    Byo�$  T          @�zᾳ33�O\)@�Q�Br�C�� ��33�8��@�{B�C��H                                    Byo��  �          @���G��q�@��HBT��C{�
��G��]p�@���Ba�Cz                                    Byo�p  �          @�
=��Q��s33@�z�BV{C|�f��Q��^�R@�33Bb�C{&f                                    Byo�  �          @�
=�c�
�s33@�
=BY�HC�n�c�
�^{@�Bg  C��                                    Byo�  �          @ٙ����
�\)@��HBO�\C|=q���
�j�H@�=qB\ffCz�=                                    Byo�b  �          @��ÿ�������@�  BK=qC�쿈���u�@�\)BXG�C~n                                    Byp  �          @�G���  ��=q@�=qBO�C�쿀  �p  @���B\=qC                                      Byp�  �          @�(�������\@��BT�C�"�����o\)@�
=Ba��C�˅                                    Byp#T  �          @����(��z�H@��Bb�RC�⏾�(��c33@ƸRBp33C���                                    Byp1�  �          @�Q�
=q���@�ffBC(�C��q�
=q���@�
=BP�C�~�                                    Byp@�  �          @�\)�aG����@��BN=qC�b��aG����@��B[��C���                                    BypOF  �          @��H<#�
��@���B1Q�C�3<#�
��33@��
B>��C�{                                    Byp]�  �          A��>����G�@���B=qC��>����\)@�z�B'�
C���                                    Bypl�  �          A��>�����@���B  C�>�>�������@�ffB��C�L�                                    Byp{8  �          Ap��L����(�@�
=B	C��\�L���ҏ\@��
Bz�C��                                    Byp��  �          A�Ϳz��ָR@��BG�C����z�����@�  B��C�n                                    Byp��  T          A�;\�ָR@��
B��C�` �\����@���B�C�L�                                    Byp�*  �          AG�������ff@�p�B{C��R��������@�=qB��C��f                                    Byp��  
�          A�;����Ӆ@�Q�Bz�C��{������G�@���B"=qC���                                    Byp�v  �          A���\)���@��RB��C�Ǯ��\)��33@��B ffC��R                                    Byp�  �          AG������p�@�\)B�
C��f����˅@��
B �C�ٚ                                    Byp��  
(          A{��R��33@���B=qC�T{��R����@�{B#
=C�1�                                    Byp�h  T          A�H�h����33@���BC��h������@�G�B$�C�޸                                    Byp�  �          A=q�Ǯ����@�p�B��C�^��Ǯ��
=@�=qB�C�K�                                    Byq�  
�          A=q�#�
���
@��B�
C�W
�#�
��=q@�
=BC�O\                                    ByqZ  
�          A�\�����(�@�G�B(�C�������=q@�ffB{C��                                    Byq+   1          Aff���
��(�@�ffB�C�� ���
�ҏ\@��B��C��\                                    Byq9�  �          A�R�n{��33@�G�B
�HC�"��n{�љ�@�ffB��C���                                    ByqHL  T          A{������G�@���B  C�G�������\)@�B�HC�\                                    ByqV�  �          A=q�^�R��z�@��RB�\C�e�^�R�ҏ\@��
B�C�<)                                    Byqe�  "          Aff�@  �ڏ\@��\BffC��)�@  �У�@��Bp�C���                                    Byqt>  
�          A�(���z�@��RB�C�y��(����H@��
B  C�\)                                    Byq��  "          A=q����Q�@�=qB�C�
=���ָR@��BG�C��{                                    Byq��  "          Aff��\)���H@�\)B
=C�� ��\)�ٙ�@��B=qC��3                                    Byq�0  �          A���R��G�@���A�Q�C����R��Q�@�
=B\)C�y�                                    Byq��  T          A�
��\)��\)@�A�  C�⏾�\)��{@�(�BG�C��
                                    Byq�|  "          A(����
��z�@�
=A��
C�����
��@�p�B=qC��                                    Byq�"  "          A�
�#�
��@���A�ffC��{�#�
��ff@�33B��C��3                                    Byq��  
�          A�����
=@\)A���C�|)����R@��RA�C�w
                                    Byq�n  �          A�Ϳ#�
�陚@��A�=qC�}q�#�
��Q�@��B
z�C�b�                                    Byq�  �          A
�H�fff��33@�G�A��C��׿fff�ᙚ@��BQ�C�^�                                    Byr�  �          A�
�W
=��33@��HB�
C����W
=��G�@���BG�C�|)                                    Byr`  
}          A(��G���{@�p�Bp�C�˅�G���z�@��B�C���                                    Byr$  	�          @�(��
=��Q�@�  A��C��f�
=��  @�p�B�C�k�                                    Byr2�  �          @�33��ff�ٙ�@z�HA��C�  ��ff��G�@�33B(�C��                                    ByrAR  �          @���>���=q@b�\A���C��=>��ڏ\@\)A�(�C��\                                    ByrO�  �          @�{>�  ��33@a�Aә�C�H>�  �ۅ@~�RA�
=C�
=                                    Byr^�  �          @��H�\����@\(�A�z�C�xR�\��G�@x��A�  C�j=                                    ByrmD  �          @�Q쾞�R���
@n�RA㙚C������R�Ӆ@��B ��C���                                    Byr{�  �          @��>u��?��HA0(�C��>u��(�?�Q�Ak�C��                                    Byr��  w          @�p�>�(���@\)A���C��H>�(����@>{A��\C���                                    Byr�6  �          AG�>����  @<��A�\)C���>���陚@[�A�33C��R                                    Byr��  �          A
=>�G�����@:=qA���C���>�G���{@Z=qAģ�C��                                    Byr��  
�          A(�?8Q����
@j=qA�ffC��?8Q���@�(�A�Q�C�޸                                    Byr�(  �          A�R?W
=����@VffA���C�9�?W
=���@u�A߮C�T{                                    Byr��  �          @�
=?��
���H@7�A��\C���?��
��(�@VffA�ffC��                                    Byr�t  �          @�?���\)@�Aw�C��
?���\@   A�C��\                                    Byr�  E          @�\)?��H����@Q�A�Q�C���?��H���
@'
=A�=qC��f                                    