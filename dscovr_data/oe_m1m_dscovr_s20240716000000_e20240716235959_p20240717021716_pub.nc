CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240716000000_e20240716235959_p20240717021716_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-17T02:17:16.994Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-16T00:00:00.000Z   time_coverage_end         2024-07-16T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy�l   T          AF=q�ָR�G����H��=qCi�=�ָR�{��  ��ffCi�3                                    By�z�  �          AB�R��
=�(���(���Q�Cg���
=�G���G���z�Ch&f                                    By��L  "          AB=q��z���\)����=qCg�=��z������
=�G�Cg�\                                    By���  �          AFff��=q����߮�	=qCf���=q�������G�Cfc�                                    By���  �          AFff��(��	G��ȣ����CiT{��(��
ff��p�����Ci�\                                    By��>  T          AC
=��G���
=�����Ce����G��������Cf:�                                    By���  �          AEp������\��G���
Cj������
��{���Ck+�                                    By�Ҋ  
�          AL  �������ٙ�� �\Cl���������{��z�ClB�                                    By��0  �          AL����{���ҏ\��Cm33��{�����
=����Cmk�                                    By���  �          AM���\)���θR���Cn�\��\)����ʏ\����Co�                                    By��|  T          AMG���Q��
=��  ��Cn����Q��Q���(���\Cn�
                                    By�"  �          AMp���
=�G��˅�홚Co(���
=��\�Ǯ��ffCoaH                                    By��  �          AN{����  �����ffCo������p�������
=Co�                                     By�*n  �          AO����H� Q���G����Cpٚ���H�!������ffCq�                                    By�9  �          AQ����H�!���33���HCp����H�#33���R��G�Cp33                                    By�G�  
�          AP  �����!p��������HCo\�����"�R���
��G�CoB�                                    By�V`  T          AN=q�����"�R��(����Co\)�����#�
��\)���Co�=                                    By�e  T          AQG���33�"ff�����33Co��33�#�
�����\)Co8R                                    By�s�  "          AN�H��ff�#�
��(���G�Cn�\��ff�$����
=��\)Cn�q                                    By��R  T          AN=q�Ӆ�%�tz���{Cmp��Ӆ�&�R�j=q��{Cm�{                                    By���  �          AN{���((��P���j�RCm�����(���Fff�^�RCm��                                    By���  �          AN=q��33�(Q��:�H�QCl�3��33�)��0���Ep�Cm\                                    By��D  �          AM�����)��>{�V{Cm�����)��333�Ip�Cm�
                                    By���  �          AO
=��\)�*{�Fff�]G�Cm���\)�*�H�:�H�Pz�Cm��                                    By�ː  T          AM���׮�*�R�Q��*�RCm�q�׮�+\)������Cm�{                                    By��6  �          AO
=��G��*�H�,���@Q�Cm�=��G��+�� ���3
=Cm�f                                    By���  
�          AO\)�����)��AG��W\)Cmp������*�R�5��I�Cm�\                                    By���  T          AN�H��ff�*{�C33�ZffCm�=��ff�*�H�7
=�L��Cm�                                    By�(  
P          AO
=���)���N�R�g33Cm�����*�\�B�\�Y�Cm�                                    By��  �          AO��ٙ��(���QG��i��Cm.�ٙ��)���Dz��[\)CmQ�                                    By�#t  
�          AO
=��Q��%��[��v=qCk����Q��&{�O\)�h  Ck��                                    By�2  T          AN�\�ڏ\�'\)�Q��k33Clٚ�ڏ\�(Q��E��\��Cm                                      By�@�  �          AN�\��(��#�
�R�\�l��Ck+���(��$���Fff�^{CkT{                                    By�Of  T          AN{��(�� ���P  �iCi�3��(��!���C33�[33Ci޸                                    By�^  �          AN�H���
�#\)�`���{�Ck
���
�$Q��S33�l��CkE                                    By�l�  T          AN�R��\�#��_\)�z�\CkO\��\�$���Q��k33Ck}q                                    By�{X  
�          AN�\���$���J�H�c�Ckc����%�=p��T  Ck��                                    By���  �          AM��Q��%p��A��ZffCkٚ��Q��&ff�4z��JffCl�                                    By���  T          AMp��ڏ\�'33�?\)�W�Cl�\�ڏ\�((��1G��G\)Cl��                                    By��J  �          AL(���
=�%���P  �k�Cl�q��
=�&�R�AG��Z�HCm+�                                    By���  
�          AK������&=q�G��c33CmaH�����'33�8���R=qCm��                                    By�Ė  �          AK�
��p��&{�J=q�ep�CmG���p��'33�:�H�T(�Cms3                                    By��<  �          AK��θR�(Q��A��\��Cnp��θR�)p��2�\�J�HCn��                                    By���  T          AK��Ϯ�)p��333�K�Cn���Ϯ�*ff�#�
�9p�Cn��                                    By���  �          AJ=q�����%G��=p��X��Cm=q�����&ff�.{�F�HCmh�                                    By��.  T          AD(���G���Mp��r�HCls3��G���H�>�R�`��Cl��                                    By��  �          AD  ������H�N{�t  Cm.�����   �>�R�ap�Cm^�                                    By�z  �          ADQ��θR�=q�Tz��{\)Cl�{�θR���E��hz�Cm
=                                    By�+   T          AEp�����G��U�{\)Ck�H�����\�E�hz�Cl
                                    By�9�  T          AF�H�������_\)��Q�Cl�������!��N�R�qG�Cm
=                                    By�Hl  �          AG����
���_\)��{Clu����
�!��O\)�p��Cl��                                    By�W  �          AG��љ�� ���Y���}p�Cl�H�љ��"{�H���ip�Cm�                                    By�e�  �          AN�R��  �%G��W
=�q�Ck�H��  �&�\�E�]�Cl
                                    By�t^  �          AN=q���H�%p��_\)�{�Cl}q���H�&�H�Mp��g
=Cl�R                                    By��  �          AM����G��"�H�\���yp�Ck\)��G��$Q��J�H�d��Ck�
                                    By���  �          APz�����%���j�H��
=Ck�)����'
=�X���qG�Cl)                                    By��P  �          AP(����#��6ff�JffCiǮ���$���#�
�5Ci�q                                    By���  �          AO\)���R�"=q���,��Ch����R�#33����  Ch�3                                    By���  T          AO����"�R�p��.�HCh�3���#�
�
�H��Ci!H                                    By��B  �          AO
=��Q��#33�(���<��Ci�f��Q��$z��ff�'
=Ci�
                                    By���  �          AL����� ���
=�)�Ci{���!��
�Q�Ci@                                     By��  "          AK\)����H�J�H�g
=Ci���� Q��7��P��Cj(�                                    By��4  �          AL(���Q��"�R�L(��g�
Ckff��Q��$(��8Q��P��Ck��                                    By��  Q          AM����
=� Q��=p��T��CiO\��
=�!�)���>=qCi��                                    By��  
�          AM����
�{�E��^ffCf� ���
���1��H(�Cg�                                    By�$&  �          ANff��
=�{�C33�ZffCg�q��
=���.�R�C�Ch@                                     By�2�  �          AM����{� ���?\)�W33Ciu���{�"{�*�H�?�Ci��                                    By�Ar  �          AL  ��  �$z��4z��Lz�Ck�R��  �%�\)�4  Ck�                                    By�P  T          AM���׮�(���7��N{CmxR�׮�*ff�!G��4��Cm��                                    By�^�  �          AM���H�*�H�333�IG�CnL����H�,(�����/\)Cn��                                    By�md  
�          AN=q��(��*ff�<���S�Cn)��(��+�
�%�9p�CnT{                                    By�|
  �          AN=q��33�*=q�C33�Z�RCn.��33�+�
�+��@Q�Cnh�                                    By���  "          ANff��
=�*�H�Mp��fffCn����
=�,���5�K�Co�                                    By��V  �          AN=q���H�'
=�*�H�?
=Ck�\���H�(z��33�$��Cl�                                    By���  T          AM�������'33�7
=�N{Cl�������(���\)�3
=Cl��                                    By���  T          AN�R�����(���7
=�L��ClǮ�����*=q�\)�1p�Cm                                    By��H  �          AO
=�����$(��
=�(  Ci�R�����%p����R�G�Ci�                                    By���  T          AN�H��z��!p��޸R��Q�Cg�q��z��"ff��\)��33Ch&f                                    By��  
�          AN�H����!G��޸R��
=Cg������"=q��{����Ch
=                                    By��:  
�          AO����\�#
=��\��=qChs3���\�$  �������
Ch�)                                    By���  "          AO33���R�!G���
=��Cg�����R�"=q�����\)Cg޸                                    By��  �          AO���p��'33��z���Cj�)��p��(Q��G���{CjǮ                                    By�,  "          ANff�����*�R������Cm)�����,  ���
��ffCmJ=                                    By�+�  �          AMp���  �*�R�
=�)G�Cm����  �,  ��Q��
�HCm�)                                    By�:x  �          AM�ڏ\�*=q�z��&{CmJ=�ڏ\�+�����\)Cm}q                                    By�I  �          ALQ�����)��{�2{Cm�����+\)��\��HCn#�                                    By�W�  
Z          ALQ����'��G��=qCl� ���(�Ϳ˅��{Cl��                                    By�fj  "          AMG���p��%녿����
Cjk���p��&�R��  ���RCj��                                    By�u  
�          AL�����
�$(����H���CjQ����
�%p����
��(�Cj��                                    By���  "          AL  ���H�#
=�  �"ffCj=q���H�$Q�����
=CjxR                                    By��\  �          AK�
��(�� z��)���@  Ci�f��(��"=q�{� ��Ci�                                    By��  T          AK���33�\)�
=�p�Ch����33� �׿�Q���(�Ch�H                                    By���  "          AI����������p����
Cf�H�����Q�O\)�l(�Cg�                                    By��N  �          AF�\���ff��ff�p�Cg�q���\)�����˅Cg�                                    By���  �          AE���\���5��S�Ci�R��\�������2�\CjJ=                                    By�ۚ  T          AE���
=�{�
�H�!��Ch0���
=�\)�޸R� ��Chp�                                    By��@  �          AF{��=q�=q�У����HCg�)��=q�\)��������Ch
=                                    By���  T          AE��������u��Cf��������������Cf�q                                    By��  �          AEG�����z῁G���
=Cf�f��������*=qCg�                                    By�2  "          AE������
=q�   Cf
����
���;�G�Cf!H                                    By�$�  �          AG
=��
=�z����!�Cf���
=��ͽ��;�ffCf#�                                    By�3~  �          AJff��\�33������\Ch�3��\� z`\)��\)Ch��                                    By�B$  T          AJ�\��G���Ϳ�33�
=qCg����G��{��
=��
=Cg                                    By�P�  "          AH(����\�=q��p���G�Cf����\�33��G����
Cg�                                    By�_p  �          AG���Q��ff��G��޸RCg33��Q��\)���
��Q�CgaH                                    By�n  �          AG\)������R�����ǮCiE�������\(��}p�Cik�                                    By�|�  �          AG������{�h����{Ch�������R��녿�33Ch�                                    By��b  �          AH����\)�녿��%Cg���\)�=q��\)����Cg�R                                    By��  T          AJ=q��=q�=q�J=q�fffCg����=q��R��zῦffCg�q                                    By���  �          AK33��R�
=�L���g
=Ce�H��R���������Ce��                                    By��T  �          AK33�=q��
�(���=p�Cf��=q�(�����(��Cf.                                    By���  T          AJ=q� ��������   CfaH� ����
�#�
�.{Cfn                                    By�Ԡ  
�          AJ{�{�33����333Cds3�{�
=>�33?�=qCdn                                    By��F  T          ALQ��p��33�Ǯ��G�Cb��p��\)=���>�Cb#�                                    By���  T          AM��	��
�L�ͿaG�CcǮ�	��
>��
?�Cc�                                    By� �  
Z          AL���z��(�>�\)?��RCe���z���?O\)@hQ�Ce�
                                    By�8  �          AL����
���>�(�?���Ce�3��
�(�?z�H@��
Ce�
                                    By��  �          AN{�=q�  ?(�@-p�CeL��=q�33?�33@�z�Ce(�                                    By�,�  	�          AN{��H�33?@  @Tz�Ce���H�=q?��@�Q�Cd�
                                    By�;*  �          AM������?n{@�z�Ce�q�����?�p�@�(�Ce�=                                    By�I�  �          AN�\��R�Q�?��@p�CeE��R��?�{@�ffCe!H                                    By�Xv  T          AO
=����
=?Tz�@k�Cd�\�����?��@�p�Cd^�                                    By�g  "          AQp��z���\?(�@,(�CeJ=�z��?�Q�@�\)Ce#�                                    By�u�  
�          AQ�����\)?B�\@Tz�Ce�)���ff?���@���Cep�                                    By��h  T          AR{�
�\���?�  @�(�Cd�{�
�\��
?˅@�{Cd\)                                    By��  T          AS��G����?�z�@���Cc���G��\)?�  @�\Cc��                                    By���  "          AU�(��  ?���@ǮCcG��(��ff@�\Az�Cb��                                    By��Z  "          ATQ��ff��?�G�@љ�Cc�\�ff��@
=A{Cc:�                                    By��   T          AT���ff��
?˅@�33Cc���ff��@�A\)Cc:�                                    By�ͦ  "          AUG��p��?�G�@ϮCd��p���
@�A=qCc�=                                    By��L  �          AS�
�z��Q�?���@�p�Cd��z��ff@p�A��Cc�R                                    By���  �          AQp�����?ٙ�@�
=Cc������
@�
A"ffCcp�                                    By���  "          AQ�(��G�?���@�
=Cc���(��33@�A*�RCc(�                                    By�>  
�          AQ�z����?�
=A\)Cc\)�z��ff@"�\A2�RCb�                                    By��  
�          AR�R����H?�{A��Cc������@�RA-Cc�                                    By�%�  �          ATz����G�?�\)A��Cdh����
=@ ��A.ffCc�q                                    By�40  �          ATQ��	��R?���A (�Ce\�	�z�@   A-�Cd�f                                    By�B�  �          AT(���
�Q�@�A�Cd(���
�@+�A:�\Cc�3                                    By�Q|  "          AUp��	G���
@	��AQ�CeQ��	G���@4z�AC
=Cd�
                                    By�`"  �          ATz��
�H���@��AQ�Cd��
�H�ff@333AB�RCd�                                    By�n�  �          AS\)�
=q��@  A�Cdc��
=q���@:=qAK�
Cc�)                                    By�}n  
�          AR�H�	��33@�A"�\CdaH�	��Q�@?\)AQCc�{                                    By��  
�          AT(��(���\@�A$z�Cc���(���@A�AS�Cc=q                                    By���  �          ATz���
�
=@{A+33Cc���
��
@H��AZ�RCcY�                                    By��`  T          AV�\�ff�
=@"�\A.�HCck��ff��
@N{A^ffCb�\                                    By��  
�          AUG�����=q@%�A2�RCck������H@P��Ab�\Cb��                                    By�Ƭ  
Z          AT��������@*�HA9�Cc�����{@W
=Aj=qCb޸                                    By��R  "          AU��{��@L(�A\��C_���{��@u�A��C^��                                    By���  
�          AV=q��\���@@  AO
=CaaH��\���@j�HA~�\C`��                                    By��  T          AV=q��R�p�@1�A?�Cc{��R�@^�RAp��Cbc�                                    By�D  T          AU�z���\@8Q�AG
=Cb0��z���R@dz�Ax  Cas3                                    By��  
�          AU���G���@7
=AEp�Ca�f�G��{@c33Av�\Ca(�                                    By��  �          AUp��33�ff@C33AS�
CbaH�33�=q@p  A���Ca�
                                    By�-6  �          ATz������H@Mp�A_�
Cas3�����\@y��A��\C`�
                                    By�;�  �          AP���(���@8��AL��Cb� �(���@eA�Ca�R                                    By�J�  �          AN�R����Q�@0��AE��CcG�����z�@^{Ayp�Cb�                                    By�Y(  
�          AM�	p��ff@1�AG�
CbǮ�	p��ff@^�RA{�Cb�                                    By�g�  T          ANff�p��G�@I��Ab�RC`�R�p����@uA��\C`\                                    By�vt  T          ANff�ff�
�H@`��A|z�C`E�ff�{@�{A�\)C_=q                                    By��  �          AK�
�
ff�
=@\��A{33Ca#��
ff�=q@�(�A�33C`!H                                    By���  T          ALz��  �Q�@G�Ab{Cb�3�  ��
@uA��Ca��                                    By��f  
�          ALQ���R��@J=qAeG�Cc!H��R�z�@xQ�A���Cb8R                                    By��  T          AM��33��@P��Ak�
Cc��33�z�@\)A�
=Cb�                                    By���  �          AM���(���
@[�Aw�Cb���(��
�H@���A��HCa�=                                    By��X  �          AN=q�Q��
�R@p  A�p�C`���Q��G�@��RA��
C_��                                    By���  �          AMG��=q�{@{�A���C_T{�=q� z�@��A��\C^
                                    By��  T          AM��=q�=q@w
=A�=qC_Y��=q� ��@���A�ffC^�                                    By��J  T          AMp����33@z=qA��C^
�����H@��\A�\)C\��                                    By��  �          AL(��� Q�@|��A�Q�C]Y������@��
A�C\�                                    By��  �          AMp���\� Q�@�=qA�  C]0���\��z�@�  A��C[�{                                    By�&<  T          AK�
����p�@�=qA��C\�R����G�@��A��RC[�{                                    By�4�  �          AJ�H�33�\)@�p�A��C_T{�33���@��
A���C]�                                    By�C�  
�          AK��(��G�@{�A�C]�f�(����R@��
A�=qC\��                                    By�R.  �          AK��p���G�@���A�{C\�
�p���(�@�
=A�  C[�                                    By�`�  �          AK33�33�   @��A��HC]Ǯ�33��33@�33A���C\Q�                                    By�oz  �          AK\)�33��{@��A��\C]�{�33���@�Q�A�p�C\�                                    By�~   "          AJ�R�(��{@�A�{C^�H�(���
=@�z�A��
C]h�                                    By���  �          AJ{�=q���H@�=qA�Q�C]n�=q��p�@���A�\)C[�)                                    By��l  �          AJ�H���\)@�G�A�=qC^�����@�  A��
C\u�                                    By��  �          AJ�R�G���p�@��A�
=C\+��G���@�G�A�CZ�=                                    By���  "          AK
=�����R@�=qA�G�C\5������@���A�=qCZ�{                                    By��^  �          AM��\��33@��A�ffCZ���\���@�G�A���CYG�                                    By��  T          AO33���  @��A��CW^������@�  A�p�CU�=                                    By��  �          AO
=�   ���@�z�A��CV@ �   ���H@�Q�A��CTaH                                    By��P  T          AM���ۅ@�\)A��CV�\�����@��A�{CT��                                    By��  "          AM��=q��(�@�G�A��
CX}q�=q���@�ffA�p�CV�f                                    By��  �          AL����
��p�@��HA��CWh���
��ff@�\)A���CU��                                    By�B  "          AN{���ᙚ@�z�A��RCW�����=q@���A�=qCV                                    By�-�  �          AO��=q��p�@�Q�A�{CW  �=q��@��A�33CU                                    By�<�  �          AO33�
=��ff@���A�  CU�R�
=��@�G�A�z�CS��                                    By�K4  �          AP  �#
=�ʏ\@�
=A�(�CS�)�#
=���@�=qA�p�CQ��                                    By�Y�  T          AN=q�"ff���
@���A���CS��"ff���H@�z�AυCP�
                                    By�h�  �          AO�
��R�ָR@���A�(�CV\��R��p�@�p�A�33CS�H                                    By�w&  
�          AJ�H�
=���@���A�(�CR�R�
=���@�
=A���CPQ�                                    By���  �          AJff� �����H@�Q�A���CR0�� ������@��\A�p�CO�
                                    By��r  �          AK33�����H@�\)A��HCRY�������@���A�CO޸                                    By��  �          AM�p���  @�\)A�  CX��p���ff@�p�A�
=CU�                                    By���  �          AO�����ff@��A�ffC[h�����@�(�A��CYxR                                    By��d  �          AR=q�{� ��@�{A��C\��{���@�  A�z�CZ�)                                    By��
  �          AS
=�=q�33@��A�z�C]��=q��ff@�z�A��C[\)                                    By�ݰ  
�          AO33�Q���Q�@��A�p�CY�)�Q��ʏ\@�Q�A�(�CW{                                    By��V  T          ALz��G���R@���A���C[=q�G����@�  A���CX��                                    By���  �          AO33�
=�޸R@��A���CY&f�
=��G�@��
A��CVh�                                    By�	�  �          AO������{@�  Aȏ\C[J=������@ȣ�A���CX��                                    By�H  �          AP���Q�����@��A�  C\Ǯ�Q���p�@���Aݙ�CZz�                                    By�&�  �          AO33�p��=q@�Q�A�Q�C_���p����@�z�A�(�C]�H                                    By�5�  "          AR�R�33�\)@�{A��HC_c��33����@��HA���C]k�                                    By�D:  "          AO33�G�� ��@�G�A��C^Q��G���@�z�A�ffC\�                                    By�R�  "          AN�H�
=q� z�@�Q�A�Q�C^��
=q����@��
A�(�C\��                                    By�a�  
�          AO33�������@�A�Q�C]�������z�@ȣ�A�p�C[\                                    By�p,  "          AM������(�@���A�ffC\Ǯ�����\)@�
=A��CZE                                    By�~�  �          AS�������H@��A���C[8R������@��
A��CX��                                    By��x  
�          AUp��
=��Q�@�ffA�ffCZ��
=��=q@�Q�A��
CWٚ                                    By��  
�          AV=q��
��p�@�33A�G�CZ���
��ff@��A�z�CW:�                                    By���  
�          AW�
�p�����@���AՅCY33�p���G�@�33A�Q�CVG�                                    By��j  T          AY�������@��A�ffCZff����Ӆ@��
B{CW=q                                    By��  T          AXz������@أ�A�{CY\)�������@�B�\CU��                                    By�ֶ  
�          AW��=q�أ�@�
=B�\CX���=q��(�@��RB��CT                                    By��\  �          AV�R����{@�G�B  CX+������A�
B�CS��                                    By��  �          AU�(���ffAz�B=qCY���(���p�A�B){CUL�                                    By��  �          AX(�������HA33BQ�C[������=qA33B%�HCW\                                    By�N  �          AZff�	���@�p�B��C[33�	����A33B�CV�q                                    By��  T          A]G��33��p�@�z�B�\CY���33��A
�\B��CU�                                    By�.�  
�          A_�
�G����A   B�CU�=�G�����A�B�CQff                                    By�=@  
�          Am��\)�I?fff@_\)Cnz���\)�F�\@�A�RCn\                                    By�K�  �          Ak��
�\�)�@l(�Ap��Cf�\�
�\�!p�@�ffA��Ce^�                                    By�Z�  �          Ap(�����4z�@���A�G�Ch�{����)@�(�A�Cg�                                    By�i2  �          Atz��	�9G�@���A�\)CiY��	�-�@��A�Q�Cg�)                                    By�w�  �          AuG��(��9�@��HA�G�Ci���(��-�@׮AиRCg�\                                    By��~  "          Au��p��@  @��RA���Ck33�p��5�@�p�A���Ci��                                    By��$  "          Aw��33�@��@�\)A���Cj���33�5�@ƸRA�p�CiaH                                    By���  �          Aw�����E�@�{A{33Cl)����;�
@�
=A�\)Cj�R                                    By��p  �          Aw33�
=�G�@�Q�Ap��Cl���
=�=@��A���CkaH                                    By��  
�          Au���z��A@�\)A��Ck�H�z��7
=@��A��Cj�                                    By�ϼ  "          A�p�� ���[�@C�
A-�Co�
� ���S�@�G�A���Cn��                                    By��b  
Z          A��\����Z{@U�A<  Cn� ����Qp�@��A�=qCmu�                                    By��  
�          A��R�p��Y�@]p�AC�Cn� �p��P��@�ffA�ffCmh�                                    By���  T          A�(��z��^=q@W
=A;�Co5��z��Up�@�z�A���Cn+�                                    By�
T  
�          A�=q����`z�@P��A6=qCp  ����W�@�=qA�
=Co                                      By��  T          A�(��
=�^�H@1G�AffCn�=�
=�W
=@��\A��RCm��                                    By�'�  T          A��
���a�@Q�A��Co����Z{@�
=Am��Cn�H                                    By�6F  �          A�
=��H�c
=@7
=A=qCp���H�[
=@�
=A��Co#�                                    By�D�  "          A������`z�@`  AACo^�����W
=@��HA��CnG�                                    By�S�  
�          A\)� ���T(�@p  AYp�Cn�q� ���J=q@�  A�33Cm�                                     By�b8  T          A}G���ff�T  @\(�AH��Co
=��ff�J�\@�ffA�G�Cm��                                    By�p�  �          A~ff��z��V�H@L��A:{Co����z��M�@�  A��\Cn}q                                    By��  T          A�(�����`��@!�A
=Cp�����X��@�p�A|Q�Cp�                                    By��*  T          A�Q���Q��l  @G�@��CrB���Q��eG�@���A]�Cq�{                                    By���  �          A�33����k�?��@�G�Cr������e�@c33AD��Cq�                                    By��v  �          A����h��?W
=@>�RCs�����d��@6ffA!CsQ�                                    By��  �          A��\�ə��l�þ�ff����Cv���ə��k�?�\)@�Q�Cvٚ                                    By���  �          A�ff���n�R?�?�\Cs(����k33@&ffA
=Cr��                                    By��h  T          A�(������h  @-p�A��Cq�������_�@�
=A���Cp��                                    By��  �          A�������p  ?��R@ڏ\Ct� �����i�@��HAbffCs�
                                    By���  �          A��R��\)�p(�@�@���CtG���\)�h��@�z�Ar=qCs��                                    By�Z  T          A�Q���\)�w
=?��H@��Cx���\)�p��@x��AW\)Cwz�                                    By�   �          A�����G��y��?�
=@�\)Cxٚ��G��s33@y��AV�\CxW
                                    By� �  T          A�����p��x��?�33@�  Cw����p��r�H@g�AFffCw
                                    By�/L  �          A��������v{?޸R@���Cu�
�����o�@|��AW�Cu@                                     By�=�  T          A���  �w�?���@�Cvh���  �qp�@l(�AIG�Cu��                                    By�L�  
�          A����{�m?�p�@�33Cq�H��{�g33@xQ�AT  Cq33                                    By�[>  T          A�{���t��?O\)@/\)CtL����p(�@C33A%p�Cs�H                                    By�i�  �          A�=q����pz�?�  @�
=Cr������j�H@\��A;�Cq��                                    By�x�  T          A�(�� ���n�\?���@���Cq�� ���h(�@qG�AL��Cq                                    By��0  �          A��R��G��v�H��
=��33Ct����G��u�?��@�33Ct��                                    By���  "          A��R�˅�|�þ��У�Cx��˅�{33?��@˅Cw�R                                    By��|  
�          A�(��˅�x�Ϳu�Q�Cw�˅�xQ�?��@�\)Cw�R                                    By��"  
�          A�����  �}p��A��#�
Czٚ��  ���ÿ���G�C{&f                                    By���  
�          A�����G����׿��
��C{�=��G���
=?n{@G�C{�{                                    By��n  �          A�p����\�
=�#33�
�\C|O\���\�����Q쾔z�C|�                                    By��            A�����z��{
=�u�T��C8R��z����R�������C��                                    By���  �          A������(�����{CzxR������?�?�z�Cz�
                                    By��`  �          A�\)��������<(���Cz����������Ϳ��C{�                                    By�  "          A���ə����\�&ff�	G�Cx�
�ə���=q��\)�uCx�
                                    By��  T          A�  ���\�����#33�=qC{0����\���\=#�
=�C{ff                                    By�(R  �          A�{��Q�����z���G�Cz�f��Q���(�?�?�p�CzǮ                                    By�6�  �          A�{��z����Ϳ��
��Cyn��z����R?���@��Cyn                                    By�E�  �          A�z������
=��G���(�Cyff�����
=?��@���Cyc�                                    By�TD  �          A�z���ff���׾���  Cxu���ff����@��@�Q�CxL�                                    By�b�  "          A�  ��(���=q����G�Cx����(���33@	��@�G�Cxn                                    By�q�  �          A����˅����G���33Cx���˅��=q@   A�
Cx^�                                    By��6  
�          A����\�
=?s33@HQ�Cv���\�yG�@`��A9G�Cu�\                                    By���  �          A�  ��Q���?   ?У�Cx&f��Q��~�H@G
=A$(�Cw�=                                    By���  
(          A�=q���H��(��.{���Cz�=���H����@!G�A(�CzT{                                    By��(  
(          A�����\)��33�#�
��C{��\)���@+�A(�Cz�=                                    By���             A��������?
=?�C{E�������@Tz�A-p�Cz�                                    By��t  �          A���ȣ���?
=q?�G�Cy#��ȣ���33@P  A)Cx�                                    By��  �          A�\)������
?G�@!G�Cy������
=@_\)A6ffCx�f                                    By���  
�          A�G����
���H?Q�@*�HCw�����
��
@a�A8(�Cw�                                    By��f  
�          A����ʏ\���������Cx���ʏ\��@-p�A��Cx��                                    By�  
�          A�
=��33��?0��@  Cy����33����@\(�A4z�CyE                                    By��  
�          A��������
=����Q�C{\)������
@�@�
=C{33                                    By�!X  T          A�
=���������@  ��C{�)�������R@�@�G�C{}q                                    By�/�  
�          A������\��{?z�?��C|\)���\��\)@Z=qA3
=C|                                    By�>�  
�          A��������\)����G�C}�������\)@6ffA��C}��                                    By�MJ  "          A��R���R����W
=�.{C~s3���R���
@,(�A��C~B�                                    By�[�  |          A�  ���\��=q>k�?@  C|����\���@C�
A#
=C|��                                    By�j�  
�          A��������Q�>B�\?�RC{G�������{@?\)A Q�Cz��                                    By�y<  @          A�G��׮��=q?O\)@+�Cw5��׮�zff@e�A=��Cv��                                    By���  "          A�������|��?�{@fffCsǮ�����u��@u�AH��Cs#�                                    By���  
�          A�p���
�w�?\(�@333Cq@ ��
�qp�@c33A9p�Cp�H                                    By��.  
�          A������R�}p�?   ?У�Ct
=���R�x  @QG�A+
=Cs��                                    By���  "          A����������H>��
?��Cx�H������Q�@L��A'�Cx@                                     By��z  
�          A�p���ff��ff>���?}p�C{�=��ff���@N{A*=qC{u�                                    By��   
(          A�(����
��  >���?}p�Czh����
��p�@N{A)��Cz\                                    By���  �          A��R������(�>B�\?!G�Cy��������@HQ�A$  Cy��                                    By��l  
Z          A��
�\���þk��E�Cy�H�\��33@,��A{Cy^�                                    By��  "          A�����p���
=��׿\Cx�)��p�����@�RA��Cxc�                                    By��  	�          A�=q��z���zᾞ�R��  Cx����z�����@(��A
�\Cx^�                                    By�^  T          A�p��ٙ���ff��{��{Cw��ٙ��}��@$z�A�CvǮ                                    By�)  T          A����33����{����Cv���33�|z�@#�
A�RCu                                    By�7�  T          A�(���33��z�\��p�Cx�q��33���H@&ffA��Cx�                                     By�FP  "          A�{�����p��5�Cy�������=q@33@�=qCy�=                                    By�T�  "          A����G����H�Q��-p�Cy�q��G����
@(�@�
=Cy��                                    By�c�  
�          A��\��  ���\��=q�c�
CxJ=��  ���R@/\)A�Cx                                      By�rB  T          A�����
=��G��B�\�!G�Cv�f��
=�~�\@333A�\CvT{                                    By���  �          A��\�����\�aG��8Q�Cv)���}G�@1G�A�Cu�=                                    By���  
�          A�����G��y녾�(���Q�Ct8R��G��v�H@\)A�
Cs�                                    By��4  �          A��
���z�H��׿�ffCu�����w�
@�RAQ�CuT{                                    By���  �          A�33��  �z=q�E��%�Cu޸��  �x  @�@陚Cu��                                    By���  �          A�������z�\�h���B�\Cu� ����x��@�
@��
Cu��                                    By��&  "          A�{��Q��y�u�J=qCu���Q��v{@.{Ap�Ct��                                    By���  �          A��R��{�|�׿O\)�.�RCw����{�z�\@��@��Cw�f                                    By��r  
�          A��H����w�
����aG�Ct�
����t  @,��A��Ct�                                     By��  "          A��������x��>�Q�?�Q�CuL������s
=@U�A333Ct��                                    By��  T          A�33�޸R�z�\?@  @\)Cv��޸R�s�@o\)AIp�Cun                                    By�d  �          A����\)�x��>��?   Cu���\)�s�@H��A(��Ct�
                                    By�"
  "          A�\)��G��x�ͽ��
���Ct�H��G��tQ�@;�AG�Ctz�                                    By�0�  �          A�(��߮�xz�<#�
<��
Cu�߮�s�@AG�A#�CuW
                                    By�?V  "          A�(��Ϯ�{�<�=�Q�Cw���Ϯ�v�H@EA'�Cw+�                                    By�M�  "          A�=q��G��}���  �Y��CxO\��G��y�@6ffA{Cw��                                    By�\�  "          A�=q���
�|�׾���{Cx����
�y�@(Q�A=qCw�q                                    By�kH  �          A�=q���H�w\)�333�ffCu^����H�t��@
=@�\)Cu#�                                    By�y�  
�          A���{�yG��!G��Q�Cv��{�vff@p�Ap�Cv��                                    By���  �          A�\)�У��yp��W
=�6ffCwL��У��w33@G�@�ffCw)                                    By��:  T          A�G���ff�y���c�
�@  Cw����ff�w�@\)@�CwY�                                    By���  
�          A�33��{�y�:�H��RCw����{�w
=@��A�RCwY�                                    By���  "          A������xQ��Ϳ���CvO\����t��@%�A�
Cv�                                    By��,  "          A�=q��  �vff�u�L��Ct�=��  �r=q@7�A
=Ctff                                    By���  T          A�33��33�w�
�����}p�Ct�)��33�s�@5AQ�Ct=q                                    By��x  	�          A��H���H�x�þ����
Cu����H�uG�@-p�A��Cu0�                                    By��  �          A����z��|  ����p�Cv\)��z��xz�@,(�A(�Cv\                                    By���  �          A�=q�����|�׾k��E�Cv  �����x  @?\)A�Cu�)                                    By�j  T          A�����H�~{��G���Q�Cu�����H�y�@H��A&�\Cu��                                    By�  T          A��
��z��z�R=�Q�>�z�Cu����z��u�@S33A0��Cu                                    By�)�  T          A�G�����zff=L��>#�
Cu�R����t��@QG�A/�
Cu=q                                    By�8\  "          A�������z�R�#�
���
Cv.����uG�@N�RA.{Cu�R                                    By�G  T          A�ff���H�y�Ǯ����CvT{���H�u@6ffA{Cu��                                    By�U�  "          A�(���33�y��u�L��CvB���33�tz�@@��A"�HCu�)                                    By�dN  
�          A������y���G���Q�Cvc�����t  @I��A*�RCu�                                    By�r�  
�          A������x�׽�G���Q�Cvk������s�@I��A+\)Cu�R                                    By���  �          A�=q�˅�|��>aG�?@  Cx��˅�v=q@b�\A?�
Cw�=                                    By��@  
Z          A�Q���=q�{
=�#�
��CwL���=q�u��@Q�A1��Cv�
                                    By���  T          A��
��{�t��?�p�@��\Ct����{�iG�@�p�A��RCs��                                    By���  T          A�G���Q��up�?�Q�@���Cus3��Q��k�
@�p�Ar=qCt��                                    By��2  "          A�z�����zff>�?�=qCx�{����s33@r�\APz�Cx@                                     By���  
�          A�  ��
=�t(�?Ǯ@���Ct����
=�i�@���A��Cs��                                    By��~  "          A�  �G��r=q?�  @UCq��G��iG�@�
=Ac�Cq�                                    By��$  "          A��R���H�t(�?���@l(�Cs�����H�j�R@�33AlQ�Cr�H                                    By���  �          A�(�����q?.{@  Cq5�����i@z�HAR{Cpc�                                    By�p  
Z          A�p�� ���rff?(�@G�Cr�� ���j�\@w�APQ�Cq=q                                    By�  "          A��
���s
=>��?���Cq�)���k�
@l(�AE�Cq#�                                    By�"�  
Z          A�(���q��=�\)>uCq���k�@W
=A3�Cph�                                    By�1b  T          A��
��
=�t(�?#�
@	��Crn��
=�l  @}p�AT��Cq��                                    By�@  "          A�33���nff>�33?�CpY����g\)@fffAB{Co�)                                    By�N�  
�          A��H����m?��@���CpǮ����c\)@���Av�HCo�                                    By�]T  
�          A�  ��
�o�@�R@�Cq0���
�a�@�\)A�(�Co��                                    By�k�  T          A�{����tz�>��H?��Crff����l��@w
=AN�HCq�H                                    By�z�  T          A��H� z��v=q=�>��Crs3� z��o�@a�A<  Cq�\                                    By��F  
Z          A�ff��
=�up�>Ǯ?��Cr�\��
=�m�@r�\AJ�RCq��                                    By���  �          A�� ���s33=�G�>\Cr�� ���l��@`  A<  Cqh�                                    By���  "          A�33� (��rff>Ǯ?���Cr)� (��j�R@qG�AK�Cq\)                                    By��8  T          A��H����p��?��\@Z=qCq�R����f�H@��
Al��Cp��                                    By���  
�          A�p�� ���r{?�ff@_\)Cq�R� ���h(�@�p�An�HCp��                                    By�҄  "          A�  ����u�>B�\?#�
Cr������n{@i��AC\)Cr�                                    By��*  �          A�p��  �p  ?���@��
Cq33�  �d��@��A�Q�Cp                                    By���  �          A��H��=q�t��?z�?�
=Cs����=q�lz�@�G�AZ�RCr��                                    By��v  
�          A�z���Q��r�\?O\)@.�RCr�f��Q��iG�@��Af�\Cq�R                                    By�  
�          A�  � Q��o\)?�G�@Z=qCq��� Q��ep�@���Ap  CpǮ                                    By��  	�          A�(����s�
?p��@J�HCt���i�@��Ap��Cs�                                    By�*h  
�          A�ff����|z�?.{@�\Cw������s\)@�G�Ai�Cw!H                                    By�9  
�          A�=q��(��|(�?h��@Dz�Cw�3��(��q�@���Av=qCw�                                    By�G�  
Z          A��\��=q�{
=?���@�(�CwB���=q�o
=@�
=A�\)Cv@                                     By�VZ  
�          A��R��p��z�\?�G�@��Cv���p��n{@�33A���Cuٚ                                    By�e   �          A����(��tz�?�\)@p��CsxR��(��i��@�z�A{�
Crk�                                    By�s�  T          A��
����s�
?�
=@�33Ct������g�
@�ffA�Cs\)                                    By��L  
�          A�����qp�?@  @"�\Cs����g�
@�Q�Ah��Cr
                                    By���  
�          A��   �o\)>��?���Cq�)�   �g33@y��AT��Cq                                    By���  	�          A�=q��H�l��>u?Q�CpL���H�d��@mp�AIp�Co}q                                    By��>  
Z          A�Q�����k�=�\)>�  Co������d��@b�\A@  Co{                                    By���  T          A�ff���jff>k�?B�\Co:����b�H@k�AG�Cnff                                    By�ˊ  T          A��
�H�iG�������Co:��
�H�c
=@UA5Cn�=                                    By��0  
�          A��	���j{��G���Q�Co�=�	���c�
@W�A7\)Cn�)                                    By���  �          A�\)��H�e>�p�?�  Cn#���H�]@q�AN�RCm8R                                    By��|  T          A�����h  ?�\?�p�Cn����_\)@}p�AX  Cm�3                                    By�"  "          A�����h  >�(�?�(�Cn������_�@x��ATQ�Cm                                    By��  �          A�������f=q>�G�?�p�Cn�{����]@xQ�AT��Cm��                                    By�#n  �          A�=q�33�b�H?�?�G�Cm� �33�Z=q@z�HAX  Cl��                                    By�2  T          A�{���b=q?\)?�z�Cm�)���Yp�@}p�AZ�\Cl��                                    By�@�  �          A�\)���d��?�z�@{�Cm�����YG�@��HA{�
Cl��                                    By�O`  �          A�(��{�`��?�G�@�G�Cm  �{�UG�@�(�A�  Ck�
                                    By�^  "          A��
=�\(�?���@�=qCl�R�
=�O�
@�Q�A�Q�Cks3                                    By�l�  �          A��R��\�^�H?��@��Cmh���\�R�R@�  A��Ck�                                    By�{R  T          A���G��\��?h��@H��Ck�3�G��R�\@�Q�Al��Cj��                                    By���  
�          A�ff�  �]G�?E�@(Q�Ck��  �S�@�z�AdQ�CjL�                                    By���  
�          A�=q�=q�^{?aG�@?\)Ck��=q�S�@�Q�Ak�Cj�f                                    By��D  
Z          A��
����]�?�33@|(�Cl.����R�\@�G�A{�Cj�                                    By���  T          A�����]��?�\)@uCl)����R=q@�Q�Az=qCj��                                    By�Đ  �          A��H�{�^�H?�(�@��Cl��{�R=q@�(�A�z�Cju�                                    By��6  T          A������`��?��@�
=Clu����S�
@��A�
=Cjٚ                                    By���  
�          A�Q��=q�\Q�?�33@�  Ck���=q�M�@���A�z�Ci��                                    By���  T          A�  ��\�Y�?��@ҏ\Cl  ��\�J�R@�
=A�Cj&f                                    By��(  �          A�
=��\�\(�?�p�@��Cl\)��\�O33@�(�A��RCj�q                                    By��  
�          A��\�z��ep�?�?�\Cn��z��\  @�33AaCmp�                                    By�t  T          A�ff�z��d��?
=@ ��Cnz��z��[33@�p�Af{Cm\)                                    By�+  �          A�=q�\)�`Q�?p��@Q�Cn#��\)�UG�@�ffAy�Cl��                                    By�9�  
Z          A�ff���_
=?z�H@X��Cm�=���S�
@�
=Az�HCl0�                                    By�Hf  
�          A�Q����^�R?��@���Cm�f���R=q@���A���Cl�                                    By�W  
�          A���=q�[�?�
=@�33Cm��=q�M��@��A��HCkL�                                    By�e�  	�          A���z��Z�R?�ff@�G�Cl���z��L(�@�\)A��Cj��                                    By�tX  �          A�����[33@�@�p�Cm:����K33@���A��Ck:�                                    By���  
�          A���
=�]�?�(�@�(�Cm��
=�Nff@��RA��RCl
=                                    By���  �          A�  �Q��]p�?�Q�@׮Cm�f�Q��N{@�p�A��Ck                                    By��J  �          A����R�_33@ff@�\Cn����R�O
=@��A��
Cl�3                                    By���  T          A���R�[�
?�{@��Cm���R�M@�33A�Q�Ck@                                     By���  
�          A����H�\��?���@��HCm{��H�O�@��HA��\Ckz�                                    By��<  T          A�(���[
=?˅@���Cl^���L��@��\A�\)Cj�
                                    By���  �          A����
�X��?�(�@��Ck����
�J=q@�A�ffCi�{                                    By��  	�          A�p����W
=?˅@���CkG����I�@���A��RCis3                                    By��.  "          A��H�p��UG�?�Q�@�Cj���p��F�R@��A��Ci�                                    By��  
�          A�
=�  �Q?�{@��Cj�=�  �C�@�\)A�  Ch�f                                    By�z  
�          A�����R{@z�@�Cj�H���B{@�{A�33Ch�R                                    By�$   T          A�=q�ff�S�@ ��A{CkW
�ff�A��@���A��Ch�                                    By�2�  "          A�p����R�R@5�A!�Ck�����?\)@�{A�{Ci�                                    By�Al  
Z          A��R�33�P��@8��A%Ck�
�33�=G�@ǮA�ffCh��                                    By�P  T          A�Q��z��N�\@<��A)p�Ck
=�z��:�R@�Q�A�ChG�                                    By�^�  �          A�Q��G��LQ�@XQ�AC
=Cj�{�G��6�R@���A�  Cg�                                     By�m^  T          A�z�����P(�@p  ATQ�Ck�����8��@�=qA�\)Cg�q                                    By�|  
�          A�{���Qp�@�  A`(�Cj�����8��@��HA��Cgc�                                    By���  "          A��R�\)�N{@���AeG�Cjk��\)�5G�@�\A�\)Cf��                                    By��P  �          A�p�����M�@tz�AZ�\Cj������5G�@�A�z�Cg=q                                    By���  T          A��H�z��M�@mp�AT��Cj�{�z��5��@�Q�A�Q�Cg�                                     By���  
�          A�������O
=@i��AM�Cj&f����7�
@߮A�{Cfٚ                                    By��B  �          A����H�R�R@���Af{Cjk���H�9�@�G�A�z�Cf�{                                    By���  
�          A�������S�@��\A|��Cj�q����8  @�\)A�Q�Cf�H                                    By��  
�          A�p���\�U@�p�Ar=qCjٚ��\�:�R@��
Aݙ�Cg�                                    By��4  
�          A������P��@��HA��HCj�q���4  A33A��Cf��                                    By���  T          A������R�H@��Ax��Cj�\����7�@���A��Cf��                                    By��  
�          A���{�S33@�z�Aep�Cj���{�9G�@�=qAׅCg                                      By�&  �          A��\��S
=@e�AG�Cj�f��;�@�G�A��Cgc�                                    By�+�  �          A�  ��H�QG�@;�A&=qCj���H�<z�@�z�A�G�Ch\                                    By�:r  T          A�  ����N�\@qG�AV�HCj������6=q@��A�G�Cg^�                                    By�I  
�          A��H��MG�@��RA}G�Cj�)��1@��A�(�Cf�f                                    By�W�  
�          A�����
�Q�@X��A?�Cj�q��
�:=q@ۅA�{Cg�=                                    By�fd  
�          A�������O�
@eAJ�HCjff����7�
@���A�p�Cg
=                                    By�u
  �          A��
�
=�O�@r�\AT��Ci��
=�6�R@�
=A��
Cfff                                    By���  
�          A����O
=@mp�APz�Ci�=���6�\@���A�CfJ=                                    By��V  T          A�
=�(��PQ�@eAJ�RCj�
�(��8Q�@��A�Q�Cg5�                                    By���  �          A��\�  �Q�@s�
AX(�Ckp��  �8  @�G�A�=qCg�                                    By���  T          A�����R�P  @�\)ApQ�Ck�\��R�4��@�A��\Cg                                    By��H  �          A�������L��@�{Ap��CkY�����2{@��HA�RCg��                                    By���  �          A�������K�@���Ahz�Cj������1p�@�{A�Cf��                                    By�۔  �          A����z��K�@�=qAi�Cj�f�z��1G�@�
=AܸRCf�{                                    By��:  
�          A��R�{�L  @�p�ApQ�Ck&f�{�1�@�\A�RCgE                                    By���  "          A����ff�K�@�\)At  Ck�ff�0z�@�z�A�\Cg{                                    By��  
(          A��R���J�H@�p�Apz�Cj����/�
@�=qA��\Cf�                                     By�,  �          A\)�
{�I�@�=qA��Ck���
{�-�@�{A���Cgs3                                    By�$�  
�          A��\�  �M�@�p�A�
=Cls3�  �/�A��A�  Ch@                                     By�3x  
�          A����
�Nff@��A�ffCl�f��
�0��A�A��Chz�                                    By�B  
�          A��
���Pz�@�p�A�Cl�R���2�RA�RA��
Ch��                                    By�P�  
�          A����(��P(�@�  A��Cm�{�(��0��A�
A��HCi8R                                    By�_j  
�          A�����
=�P  @��RA�
=Cnz���
=�/�A33B�Cj�                                    By�n  
�          A�����
=�T��@��A��Co�H��
=�5A  A�  Ck�=                                    By�|�  T          A�����
�QG�@�{A�=qCt����
�&ffA-�B(  Co�=                                    By��\  "          A�33��{�N�R@��
A�\)CuL���{�#33A/�
B+�
Co�                                    By��  	�          A�����\�V�\@�(�A�z�Cv����\�)p�A6=qB-33Cq.                                    By���  "          A�\)���H�VffA��A�
=CxE���H�'33A<��B4z�Cr�                                    By��N  
�          A��\��G��Qp�A�B �RCx���G���
AC�B>  Cs@                                     By���  �          A�\)�����Q�A33A�
=Cx!H�����#33A:=qB5��Cr��                                    By�Ԛ  "          A������S�@�{A�G�Cu�������'
=A2�HB+z�Cp�                                    By��@  
(          A��R��z��V�RA ��A�\Cx)��z��(Q�A9G�B1��Cr�)                                    By���  T          A��H����S�A	�A��\Cx^�����"�RA@z�B9Cr��                                    By� �  
�          A�=q����P��Az�B �Cx�=�����HAC
=B>G�Cs                                      By�2  "          A�ff���R�O�A  B�RCy
���R���AF{BA�Cs+�                                    By��  T          A�(������O�A��B�RCx� �����p�AD  B?�Crٚ                                    By�,~  	�          A�  ��p��H(�AQ�B�Cz���p��AO�
BPG�Ct)                                    By�;$  
�          A�z���\)�N{A	G�A���Cv���\)���A?�B:  Co�R                                    By�I�  �          A����G��L  A=qA��Ct�f��G���A<  B6��Cn.                                    By�Xp  T          A��\�����L(�@ҏ\A��Cn�������$Q�A z�Bz�Ch��                                    By�g  	`          A�����(��K�
@�\)A�{Csc���(��G�A5�B0�Cm                                    By�u�  
(          A�������Pz�@�33A�Q�Cq������'
=A&{B�HCk�                                    By��b  
�          A����z��Q@�z�A�Cs&f��z��%��A.�HB&�\CmQ�                                    By��  T          A��������P��@��
A�z�Csff�����#�A2{B*(�Cmh�                                    By���  �          A�G���
=�O
=@�ffA��
CtO\��
=� (�A6�RB0ffCn�                                    By��T  �          A�����Q��O
=@��A�z�Ct.��Q�� ��A3�B-��Cn�                                    By���  T          A��H��z��O�@���A��Cs���z��!�A2�\B+��Cm�R                                    By�͠  �          A���ƸR�Pz�@��A�\Ct� �ƸR�!��A6{B/33Cnh�                                    By��F  �          A�����z��P  @�
=A�  Ct����z�� ��A7�B1�Cn�=                                    By���  
�          A����Q��P  AG�A�p�Cu.��Q��   A9p�B333Cn��                                    By���  
�          A�33��33�N�R@��
A�33Cs�{��33��A5B/ffCm��                                    By�8  �          A�p��Ϯ�Nff@��\A�CsE�Ϯ�\)A5G�B.p�Cl��                                    By��  
�          A�������M�@��
A�RCr������{A5��B.�Cl�                                    By�%�  "          A���=q�N{@���A��Cr�3��=q��\A6ffB/{Cls3                                    By�4*  "          A������N=q@���A�G�Cr�����\)A4z�B-{Cl=q                                    By�B�  T          A����У��N�\@��\A㙚Cs0��У��33A5��B.��Cl                                    By�Qv  �          A����˅�O33@�z�A�G�Cs�{�˅��A6�RB/��Cms3                                    By�`  �          A�G��ʏ\�PQ�@�
=A�ffCt��ʏ\�!p�A4��B-�
Cm�H                                    By�n�  
�          A����{�O�@�
=A�  Ctz���{�\)A8Q�B1��Cn�                                    By�}h  T          A�33��(��P��@�  A��
Ct�)��(��!A5p�B/=qCn�                                     By��  
�          A�{����M�A��A��Cu=q����{A@��B;G�Cnp�                                    By���  
�          A�  �����MAffA��HCt������\)A>�\B8��Cn0�                                    By��Z  �          A�\)����Mp�A�RA��Ctc�����  A;
=B5�Cm��                                    By��   "          A����Ǯ�F�\A	p�A��
CsL��Ǯ�33A?\)B<Q�Ck�)                                    By�Ʀ  	�          A�����H�C�
A��B(�Cuu����H���AJ{BI�HCm�
                                    By��L  "          A�����
�F=qA\)B��Ct�H���
���AE�BC  Cm�                                    By���  T          A�
=��\)�K\)A�
B
=Cv����\)��HAC\)B@�HCo�{                                    By��  �          A�����
=�@Q�AG�B(�Cs�)��
=�	p�AH��BH�HCk5�                                    By�>  T          A�������<��A33BffCrp�����p�AI�BJG�Ci��                                    By��  �          A�33���<(�AB��Cr@ ����AL(�BL\)Ci
                                    By��  T          A�p����@(�A�B  Cr�=�����AIBHffCj&f                                    By�-0  �          A���\�H��A�
B =qCt.�\�(�AB�HB>�Cl�R                                    By�;�  �          A���{�:=qAp�B��Cq����{� z�AO33BO�RCh^�                                    By�J|  �          A�(���z��6=qA!�B
=Cp�R��z����RAR=qBRCf\)                                    By�Y"  T          A��
�أ��0  A#\)B\)Cnc��أ��陚AQBS33Cc+�                                    By�g�  �          A����z��>ffA�RB��Cpٚ��z���AF�\BD33Cg�                                    By�vn  T          A���أ��?\)A�B�CpxR�أ����AEG�BA��Cg�H                                    By��  "          A�\)��
=�K
=A��A��RCt�{��
=��\A@��B=
=Cm��                                    By���  �          A������S33@�p�A�\)Cz������!p�A:{B8�Ct�\                                    By��`  �          A�
=�/\)�`  @�{A�ffC�w
�/\)�/�A6�HB5
=C�                                      By��  �          A�p��/\)�g�@�=qA�z�C��H�/\)�;33A,(�B'Q�C�ff                                    By���  
�          A�ff��Q��d��@\A�{C~Q���Q��;
=A#�
B�\C{�                                    By��R  R          A33��p��^�\@�(�A�\)C{n��p��4��A"�RB  Cw��                                    By���  $          A�  ����[�@�=qA��Cz  ����/�A(z�B$��Cu�=                                    By��  �          A�����
=�ap�@ǮA�=qC{xR��
=�6�RA%p�B��Cw��                                    By��D  T          A�
�����`��@�ffA��HC{@ �����7�A ��B  Cwn                                    By��  �          A~=q���\�\��@�=qA��
C{�����\�1�A%G�B"�Cw��                                    By��  
�          A~�\��{�]p�@�p�A��RC|5���{�1�A'33B$z�Cx:�                                    By�&6  "          A���=q�W�
@��HA���CzY���=q�)�A/�B-�\Cu�                                     By�4�  �          A�����U�@��A��Cy�����%p�A2=qB0�HCt��                                    By�C�  "          A~�R��ff�U�@�=qA��Cz�\��ff�%G�A2ffB2  Cu��                                    By�R(  	�          A33���\�S
=@�  A��HCy�f���\�"=qA4��B4�\Ct��                                    By�`�  
�          A~�\���\�S�@��A�p�Cx�����\�#�
A1G�B0p�Cs�{                                    By�ot  
�          A
=����T��@�=qA���Cy�����$��A2�\B1�RCt�R                                    By�~  
�          A���33�T��@�z�A�z�Cy����33�$Q�A3�B2�Ct��                                    By���  
�          A~�H��
=�S�@�Q�A��HCz^���
=�"�\A5�B5(�Cu
                                    By��f  �          A\)��Q��T��@�=qA�RC{@ ��Q��#33A6�\B6�\Cv!H                                    By��  
�          A~ff���H�U�@�(�A�G�C{  ���H�$z�A3�
B3�
Cu�3                                    By���  
�          A~�H��G��[33@��AͮC|����G��,z�A.�RB,��CxJ=                                    By��X  "          A}������P(�@�33A��Cx�H������\A5��B6z�Cs�                                    By���  
�          A}����=q�?\)Ap�A�G�Cq33��=q�  A7�B9�Ci{                                    By��  R          A}G����H�<  AG�A�  Cp�3���H�\)A:ffB=p�Ch
                                    By��J  
\          A}�����<(�A33B �HCqQ����
=A<Q�B?�
Ch�                                    By��  �          A}����  �<��Az�BG�Cr)��  �33A=BA�HCi�                                    By��  "          A|������=A�HB �Cr������z�A<��B@��Cj(�                                    By�<  T          A|����=q�>=qA�HB  Cr����=q���A<��BAffCj�f                                    By�-�  T          A|(�����?33AffB �Cs������	A<��BA�Ck��                                    By�<�  �          Az�R��ff�>{A�B{Ct\)��ff�Q�A=��BD�\Cl5�                                    By�K.  T          Ay�љ��0  A�\B\)Co:��љ���G�A?�BH�Ce
=                                    By�Y�  
�          Ax���љ��2�HA	G�B
=Co�H�љ���=qA;�BC�Cf�                                    By�hz  �          Ax����(��2ffAQ�B	G�Cp8R��(���
=A>ffBG��Cfp�                                    By�w   "          AyG���
=�6{A
=qB��CqW
��
=��\)A=��BF
=Ch�                                    By���  
�          Ax�����H�+
=A\)Bp�Ck�����H��(�A7\)B?  Ca(�                                    By��l  S          Ax���(�� z�A��B�\Cf���(���
=A5��B;Q�C[!H                                    By��  "          Ax(���Q��$��A
{B33Ci  ��Q���{A8  B?ffC]��                                    By���  
�          Av�H��33�&�HA��B�Cl����33��A?33BK=qCaL�                                    By��^  �          Aw
=��ff�+�
A	B��Cm{��ff��A:{BC�
Cb�H                                    By��  
�          Av�H����*�\A
�\B��Cl�H������A:�\BDQ�Ca�q                                    By�ݪ  
�          AuG������ z�A�HB�Ci�
�����ҏ\A;\)BG(�C]��                                    By��P  �          Aq���
=�$  A	p�B
�Ck�=��
=��z�A7\)BE�
C`��                                    By���  T          Aq��ۅ�#�A
=B�
Cl+��ۅ���HA8��BH{C`�                                    By�	�  �          Ap����p��\)AffB{Ck5���p���Q�A:�RBKQ�C_B�                                    By�B  �          Ao���33��
A��B(�Ck�{��33��=qA9��BJ��C_��                                    By�&�  
�          An�\�Ӆ�!p�AQ�B
=Cl�=�Ӆ��p�A9G�BK�HCaG�                                    By�5�  �          An�H�׮�p�A�B��Ck���׮�˅A;
=BNp�C_\)                                    By�D4  
�          Am�Ӆ�"�HA��B�\Cm�Ӆ��=qA6�\BH�HCa�                                    By�R�  �          Al����p��A	G�B�
Cj����p���  A5p�BH(�C_0�                                    By�a�  T          An=q�У��%G�A  BG�Cm���У���
=A6�RBHffCb��                                    By�p&  "          Am����&�RA�Bz�Clٚ�����A1p�BA{Cb�                                    By�~�  �          Am��G��%A\)B33Cl���G����HA2�\BB�RCb=q                                    By��r  T          Amp���\)�"�RA\)B{Cl����\)��=qA5p�BG33Cac�                                    By��  �          Ao
=���%A�B�Cl8R����=qA2�RBA�
Ca��                                    By���  �          AqG���\�-p�@�  A�Ck���\��33A&�\B.�Cb�                                    By��d  
�          Apz�����+�@�\)A��HCkY�������A)��B3ffCa�                                    By��
  	�          Ap����ff�+\)@�A��RCm���ff��A0Q�B<��Cc#�                                    By�ְ  �          AqG���Q��*�\A�
B�Cm����Q����HA4��BB(�CcW
                                    By��V  S          Ao�
���"�HA��B
=Ck������A6�HBFC`c�                                    By���  U          Ap�������$Q�@��
A��RCiT{�������HA-G�B8Q�C^��                                    By��  
�          Ap������$��AB\)CjB��������A0��B={C_}q                                    By�H  
Z          AqG����$��A33B�
Cjff���߮A2ffB>�RC_�                                     By��  �          AqG���  �"=qA�B�RCjh���  ��Q�A5��BCp�C^�R                                    By�.�  
�          Ao�
����!�A��B�CjL�����ٙ�A3
=BAffC_�                                    By�=:  "          Ao���ff�Q�AffB��Ck����ff��A=�BQ�HC^�                                    By�K�  "          Ao���ff�z�AffB��Ck����ff��A>{BQ��C^�                                    By�Z�  T          Ao���p����A=qBffCk����p���
=A>{BR  C^�q                                    By�i,  T          Ap(���  �AG�B�
Ck����  ����A=p�BPQ�C^�                                    By�w�  �          ApQ��������A�
B{Cj�������ə�A<  BN�C^^�                                    By��x  �          Ap  ���(�Ap�BffCi������Q�A9G�BJG�C]�                                    By��  �          Ap  ��(��(�A{B33Ci޸��(��ǮA:{BKG�C]8R                                    By���  "          Ao\)��\��\A�Bp�Ci���\�ÅA:�HBMQ�C\��                                    By��j  
�          An=q��G���A�B(�Ci����G�����A:ffBN  C\�{                                    By��  �          Anff��(��(�A�HB�Cj���(���z�A=G�BR33C\�{                                    By�϶  "          An{�׮�z�A�
Bp�Cj���׮��z�A>=qBT�C]#�                                    By��\  "          Amp��ָR�ffA�B�Cjs3�ָR��\)A>�HBU��C\�                                     By��  �          Amp��ٙ����ABffCi�\�ٙ���(�A>�HBV
=C[�
                                    By���  
�          Am�������{AB\)Chn������
=A>{BT��CY��                                    By�
N  "          Am���=q��A��B��Ch:���=q��
=A=G�BS�RCY�                                     By��  "          Al�����
��A�BG�Ch����
��  A<  BRG�CY�                                    By�'�  T          Ak
=��G��G�A�\B=qCh:���G���\)A:�RBRz�CY�                                    By�6@  
�          Ak���G��z�A  B�
Ch\��G���z�A<  BS�
CYxR                                    By�D�  �          Ak����H��HA�B  Cg�����H����A<z�BTffCX�3                                    By�S�  
Z          Aj�R��\�z�A=qB33Cg����\��A:ffBR(�CYxR                                    By�b2  �          Ai����H���A��BCg����H��\)A8��BP�HCY��                                    By�p�  T          Aj{��Q���\A(�B��Ch�\��Q����HA8��BP�
CZ��                                    By�~  �          Ah����
=�
=A{BG�Ch����
=���A733BO�\C[�                                    By��$  "          Ai���p��	�A�HB�CeE��p�����A8��BP��CV{                                    By���  
�          Aj=q��R�{AffB��CdL���R��
=A:�RBS�\CTL�                                    By��p  T          Ai���Q��ffA�B\)Cd5���Q���Q�A9BR(�CTc�                                    By��  
�          Ah������A��B��Cdff�����A9G�BR��CT��                                    By�ȼ  �          Ah������Q�Ap�B�Cc�q�����(�A9G�BR�RCS�f                                    By��b  T          Ah  ��=q�ffA��B��Cc)��=q����A8  BR�CR�H                                    By��  �          Ag33��
=��HA(�B�\Cc����
=��=qA7�BR��CSs3                                    By���  �          Ah  ��
=��A�B(�Cd{��
=���RA8  BQ�CT8R                                    By�T  �          Ah  ��\)��A�B{Cd
=��\)���RA8  BQ�
CT.                                    By��  "          Ag\)��G��  A�HB�Cc�{��G�����A6�HBP��CS��                                    By� �  T          Ag33��p���
A��B��Cc����p���33A8��BS�\CS�=                                    By�/F  
�          Af�\�p���
=A  B$�HC]�q�p��dz�A6{BP��CK�{                                    By�=�  T          Ag33���
���
A��B&�RC`�����
�z=qA9BVG�CO.                                    By�L�  
�          Af�\� �����A33B$  C^B�� ���j�HA5�BP�CL�=                                    By�[8  �          Ae�������z�A�B%�\CaQ�����~�RA8  BU��CO�                                     By�i�  T          Ad(���p�����A��B((�Cch���p����HA:{BZ�CQ��                                    By�x�  �          Ac33��p����HA�B*��Cb�)��p��w�A:{B[�
CPY�                                    By��*  
(          Abff������A�HB,�RCc�{���y��A;33B_=qCQff                                    By���  �          A`���׮��{A{B-{Cd�q�׮�}p�A:�RB`��CRk�                                    By��v  T          A`��������33AffB-�HCdG������w�A:�\B`�
CQ��                                    By��  T          Aap���{��A�\B-33Cck���{�s�
A:ffB_=qCP��                                    By���  "          Aa��ٙ���(�A�
B.�CdE�ٙ��w
=A<(�Baz�CQ��                                    By��h  �          A`  ��z����A��B,(�Cc����z��w
=A8��B^��CQB�                                    By��  "          A`Q������Q�Az�B+Q�Cb�����tz�A8Q�B]  CP�=                                    By���  
�          A_����
��(�A  B+G�Ca�q���
�l��A733B[�
COn                                    By��Z  �          A^�R������Az�B'(�Ca������s33A4  BW�CO�)                                    By�   T          A^�R��R��=qA{B)�\C_s3��R�\��A3�BW(�CL��                                    By��  
�          A]���p���=qA��B-C^����p��J=qA4��BY�HCK
=                                    By�(L  
�          A]������A��B/G�C\�R���8��A4(�BYG�CH�3                                    By�6�  "          A]����
=��\)A�RB1
=CZ���
=�#�
A3\)BX33CF^�                                    By�E�  T          A]p���p���\)A
=B1C[���p��#33A3�BY{CFh�                                    By�T>  �          A\����
=�ٙ�AffB+�RC^L���
=�K�A2ffBWCK{                                    By�b�  
�          A\����\)�ٙ�A{B+p�C^E��\)�L(�A2{BWz�CK
                                    By�q�  
�          A[\)�����Q�A
=B-��C\������9��A1��BX  CI{                                    By��0  "          A[\)�����A{B,��C\�q���=p�A0��BV�CIn                                    By���  
�          A[\)���R�ȣ�A
=B.{C[����R�*�HA0Q�BU�
CG
                                    By��|  
�          A\�������p�A�B3
=CWc����� ��A1�BUffCB#�                                    By��"  T          A]G�� ����(�A#�B?�CQ�f� �ͿY��A333BX�\C:
=                                    By���  "          A^{�������
A(�B/=qC_ٚ�����L��A4��B\�CL!H                                    By��n  
Z          Ad����G�� Q�A�
BCts3��G���A:=qB]�RCiO\                                    By��  �          AhQ��/\)�?�@�\)B �RC��\�/\)�	A5BN  C|Y�                                    By��  �          Aip���\�C�@�z�A��\C�����\��A5��BLp�C��                                    By��`  �          Ai��� ���?\)@��RBQ�C��� ���Q�A9�BR
=C}��                                    By�  �          Am��=p��=�AB	z�C���=p����A>�RBV=qCzQ�                                    By��  �          Am���8Q��=p�A�B33C�'��8Q���A@(�BX�Cz�                                    By�!R  �          AmG��9���;\)A	�BG�C�
=�9��� ��AAB[
=Cz.                                    By�/�  �          AmG��L(��:{A
{B\)C~��L(���
=AAp�BZz�Cx5�                                    By�>�  �          Am��`���7�A
�HB�C|�q�`�����AAp�BZ�RCu��                                    By�MD  �          AmG��g��6=qAz�B(�C|c��g���ffABffB\  Ct�{                                    By�[�  �          Am��fff�6�HA��BG�C|�=�fff��
=AC
=B\33Cu�                                    By�j�  �          Am��`  �7
=Ap�B��C|�R�`  ��
=AC�B]  Cu�
                                    By�y6  �          Anff�QG��9�A�BffC~B��QG���AB�HB[\)Cw�\                                    By���  �          Am��Y���3
=A�RBp�C}��Y�����
AG33Bc��CuE                                    By���  �          AmG��j�H�*�HAB!��C{��j�H�׮AK
=BkQ�Cqk�                                    By��(  �          Am���j=q�'\)AB&��Cz���j=q��ffAMp�Bp  Cpc�                                    By���  �          Am���w
=�$��A�B(��Cyu��w
=�ȣ�ANffBq
=Cn^�                                    By��t  �          Amp���G��"{A ��B*Cx:���G���=qAN�RBrG�ClO\                                    By��  T          Am��qG��(��A33B#��CzY��qG����HAK�Bl�
Cp:�                                    By���  �          AmG��^{�4��A
=Bp�C|��^{��ADQ�B_��CuT{                                    By��f  �          Am�\(��2�HA�HB�\C|޸�\(���33AG\)Bc��Ct�f                                    By��  �          Am��Z�H�1p�Az�B�\C|�)�Z�H��AHQ�Be��Ct��                                    By��  �          Am��x���'�Az�B$��Cy�
�x����\)ALQ�Bmz�Co                                      By�X  T          Am����"=qA
=B'��Cvk������
AL��Bm��Cj)                                    By�(�  T          Anff�����!�Ap�B%�Ct���������HAK33Bj\)Cg��                                    By�7�  �          Am���(���HA Q�B)��Cu���(���z�AM�BnCgٚ                                    By�FJ  �          Al������\)A"�\B-
=Cs�q�����(�AM�Bp��Ce��                                    By�T�  �          AlQ����H�'�Ap�B��CvQ����H��z�AEBcz�CkQ�                                    By�c�  �          Ak���z����A$��B1z�Co8R��z���
=ALQ�Bp��C^�                                    By�r<  �          Ai������G�A#33B0��Cmh��������AIBn(�C\c�                                    By���  �          Ahz��O\)�8z�A�HB	G�C~Q��O\)� z�A:{BU\)Cx�                                    By���  T          Ah���/\)�Lz�@��
A��
C��{�/\)��A%B5�
C~n                                    By��.  �          Ahz��?\)�@(�@�
=A�
=C���?\)��A2{BI�C{{                                    By���  �          Ah���~�R�<��@�=qA�z�C{Y��~�R��A2ffBG�
Ct޸                                    By��z  �          Ak\)�|���.�HA�
B��Cz)�|����{AC
=B`G�Cq.                                    By��   �          Alz���G��0��A�RB��Cy�3��G���\AB�RB]�HCq+�                                    By���  �          Af�R�hQ��.{AQ�B��C{���hQ���
=A?�B_�CsL�                                    By��l  �          Ae��`���.�RA
ffB�C|(��`����G�A=�B^��CtG�                                    By��  �          Ag�
��Q��   A\)B#{Cu����Q����AD��Bh��Ci                                    By��  �          Ah�������{AffB+�Cq�
������AH(�Bm��Cc8R                                    By�^  �          Ag\)����{A!G�B0�Cn�)�����z�AH(�Bop�C^^�                                    By�"  �          Ag�
��G��33A#�
B3��Ck�\��G���AH(�Bo
=CYk�                                    By�0�  �          Ah(���p���\)A&�HB7��ChG���p��{�AHz�Bn�
CT}q                                    By�?P  �          Ag�
��(���
A#�B3�Cnc���(����RAI��BqC]5�                                    By�M�  �          Ajff��Q��  A*ffB:�Ck{��Q����AMG�Bt�CW�                                     By�\�  �          Aj{������z�A+�B<�RCh�������o\)ALz�Bs��CS                                    By�kB  �          Aj�\��  ��\)A,(�B=
=Cg���  �eAL(�Br(�CQٚ                                    By�y�  �          Ak\)��33��G�A1G�BC��Cd���33�C�
AN�\BuffCM�q                                    By���  T          Ak
=��z���Q�A0��BCffCd����z��B�\AN{Bt�HCMn                                    By��4  �          Ai��ff��=qA0��BCCg�{��ff�UAO�By
=CQG�                                    By���  �          Adz���Q�����A1BL�\C_aH��Q��p�AH��Bv�CD�3                                    By���  �          Ad�������ffA1G�BK{C](���녿�AG
=Bq�CB��                                    By��&  �          Ac�
��p���33A0  BJz�C`�f��p��p�AH(�Bu��CF��                                    By���  T          Ac
=��=q����A2�\BO��C_� ��=q���AH��By(�CD�=                                    By��r  �          Ad(���\)��G�A2=qBM�C\�R��\)��33AG
=Bs�\CA�q                                    By��  �          Ad��������z�A6{BR�CZxR���ÿ�
=AH��Bt�C=��                                    By���  �          Ad�����
��G�A6{BR�
CY�{���
���AH  Bs��C<�q                                    By�d  �          Ad����
=����A4��BQ33CY(���
=��{AF�HBq�RC=�                                    By�
  T          Ac�
��(���33A4Q�BQ�CY޸��(���Q�AF�\Br��C=��                                    By�)�  T          Ac\)��=q���
A4  BQG�CZ33��=q��(�AFffBs33C>)                                    By�8V  �          Aap���(����A2ffBQffCXٚ��(���ffAC�
Bqz�C<��                                    By�F�  
�          Aa�������A1�BP��CX��������AC\)BpC<�q                                    By�U�  �          A`����\)���A2{BQffCW����\)�h��AB�RBo�RC;ff                                    By�dH  �          A`����
=����A3�
BT=qCVp���
=�+�AC33Bp�C9}q                                    By�r�  �          A_�
����G�A3\)BU�CR�����aG�A@  Blp�C5Ǯ                                    By���  �          A]G��߮��{A1G�BT�HCTp��߮���A>�HBn\)C7\)                                    By��:  �          A[33��  �j�HA0Q�BVffCN�)��  >�z�A9Bh  C1�                                    By���  �          A[\)����AG�A0��BVffCI�����?h��A6�\Ba(�C-#�                                    By���  �          AY�����FffA.ffBT�
CJaH����?G�A4��B`�C.\                                    By��,  �          AY����(��UA,  BP�CK����(�?�\A3�
B_�C0+�                                    By���  �          AW����R�w
=A&{BI{CN�
���R��G�A1�B\�C4�\                                    By��x  �          AW33���
�U�?��@�G�C��{���
�Ap�@�\)A���C��3                                    By��  �          A[\)��z��Y�?�z�@��C�b���z��Ep�@�ffA��
C�Q�                                    By���  �          A^�R�xQ��\z�?ٙ�@�  C��q�xQ��F=q@ȣ�A�{C���                                    By�j  �          A^�\�����[�
?�p�@�z�C�������Ep�@���A���C�b�                                    By�  �          A]p��E��\  ?��\@���C�e�E��H(�@�33A�Q�C�<)                                    By�"�  �          A\�׿&ff�[�?�  @�
=C����&ff�G�@�=qA�C��                                     By�1\  �          AX(���G��S�
@�A��C��)��G��<(�@�A��
C�T{                                    By�@  �          AZ�\����U�@ ��A)G�C�z����;
=@��
A�p�C���                                    By�N�  �          AY��ff�Mp�@dz�At��C��\�ff�.{@�ffB
�RC��f                                    By�]N  �          AR�\�~{�\)@��HB(�Cw��~{�ϮA*�\BZ  Cn�\                                    By�k�  �          AP����  �&�R@�\)A��RCx�q��  ��G�ABDp�Cr                                    By�z�  �          AT(��tz��6�H@�
=A�p�C{���tz���\A(�B+��Cv�=                                    By��@  �          AX����\)�
=A	p�B��Csc���\)���A4Q�B_��Cg�3                                    By���  �          A\(������RA�
B�
Cru������p�A6�\B`
=CfO\                                    By���  �          A\z����
��A\)B({Cq�=���
���A;33Bh��Cc��                                    By��2  �          A]���p���AB)Cq=q��p�����A=G�Bj=qCc
                                    By���  T          A]������HAp�B)ffCp����������A<��Bi\)Cbc�                                    By��~  �          A]p���{�Q�A  B�
Cr�
��{����A7
=B_  Cf��                                    By��$  �          A]����=q�Q�A��B#�CoxR��=q��ffA9�Bb�\Ca�3                                    By���  �          A^�R��ff�p�A
ffB��Cq����ff��(�A5�BZ�Ce��                                    By��p  �          A_\)���� Q�A��BG�Cs������z�A3�BVCi�=                                    By�  �          Aap����\�0��@��B33Cy�q���\��=qA0  BM�Crp�                                    By��  �          A`���j�H�5p�@�RB �C|{�j�H�
=A-G�BIp�Cu޸                                    By�*b  �          Ab�H�1��?
=@��
A�G�C�t{�1��A+\)BDG�C|�)                                    By�9  �          Ab�R�;��?
=@�Q�A�C�q�;���\A)BB{C{�\                                    By�G�  �          Ab{�L(��9�@��
A��
C~���L(���A-p�BHffCyaH                                    By�VT  �          Ab=q�l���6=q@���B �\C{�q�l����A.ffBI��Cu�q                                    By�d�  �          Ab{�h���2�R@�33B  C{��h������A2{BO�CuB�                                    By�s�  
�          Aap��hQ��-��A\)B�RC{���hQ���\)A5BW=qCt#�                                    By��F  �          Aa���p  �*{A�B�RCz���p  ��{A8z�B[�Crp�                                    By���  �          Aa���z��'
=A�B  Cx^���z����A7\)BZQ�Coz�                                    By���  �          A`Q����{AffB�RCu�
����33A:�HBa�HCk{                                    By��8  �          A`z�����%p�AQ�Bz�Cx)�����p�A7�B[z�Co                                      By���  �          A`(���=q�#\)A	�B�
Cw\��=q�أ�A7�B\
=CmxR                                    By�˄  �          A`�����
�#�
A	p�B�\Cv�)���
�ٙ�A8  B[�\Cm@                                     By��*  �          A`����=q�'
=AB  Cw}q��=q��=qA5��BWz�Cn�=                                    By���  �          A`(����\�!�A
=qB�\Cv�\���\��A8(�B]\)Cm�                                    By��v  �          A_���\)��A�B�Cu����\)��z�A9p�B`=qCj�                                    By�  �          A^{���\�,��@�p�B\)CyW
���\����A-�BM�Cq�R                                    By��  T          A_\)��p��*=q@�
=B�\Cx�)��p�����A0��BQCp�)                                    By�#h  
�          A`  �����A(�B��Cu�������Q�A8��B^�CkaH                                    By�2  �          A_����"�RA�B��Cvs3����G�A5BY�Cl�H                                    By�@�  �          A^ff���R� Q�Az�Bp�Cv���R��z�A5B[ffCl#�                                    By�OZ  �          A^=q���R�=qA
�RBp�Cu�����R��\)A733B^  Cku�                                    By�^   �          A^=q�|(���
Az�B{Cx}q�|(���G�A9p�Bb�Cn�                                    By�l�  �          A^ff�}p��"=qA
=qB�RCx�3�}p���\)A8  B_Q�Co�\                                    By�{L  �          A^ff��33� ��A
=BCw����33��(�A8Q�B_�CnE                                    By���  �          A^ff��=q� Q�A�B��Cw�f��=q���HA8z�B`�CnQ�                                    By���  �          A]�z�H��A��Bp�Cx�\�z�H����A9G�Bb��Co�                                    By��>  �          A]��{��z�A
=B!=qCx)�{��ə�A:ffBe��Cn\                                    By���  �          A\����(���A	G�B�Cwz���(��ӅA6{B^\)Cm��                                    By�Ċ  T          A\����=q� ��AffB{Cv��=q��  A3�BZ
=CmaH                                    By��0  �          A\������ ��A(�B\)Cw�����ָRA5G�B\�HCn��                                    By���  �          A^=q�u�'\)A(�B33Cy�
�u��A3�BX�Cq޸                                    By��|  �          A]����(��G�A��B$ffCv����(��\A;�
Bg�RCk�=                                    By��"  �          A\����
=�33A��B/�HCt���
=��33A?
=Bq=qCg�R                                    By��  �          A_33���
�!G�A33B�Cw���
��{A8(�B^�RCn^�                                    By�n  �          A[���{� (�A=qB��CwJ=��{��\)A3
=BZ�RCn�                                    By�+  �          AZ�H�n�R�%�A{B�Cz)�n�R��A0��BW�CrL�                                    By�9�  T          A[\)�G
=�9�@�=qA��C�G
=���A�
B;33Cz�f                                    By�H`  �          A\  �XQ��9@У�A���C}Ǯ�XQ��A
=B9ffCy!H                                    By�W  �          A\(���z��)��@�\B��Cw}q��z����A)BI��Co޸                                    By�e�  �          A\  ��  ��HAQ�B{Cv����  ��(�A4z�B\G�CmT{                                    By�tR  �          A[��x����
A��B �\Cx=q�x����33A8Q�Bdz�Cn��                                    By���  �          AZ�R�q��p�A�B$=qCx}q�q���p�A9��Bh(�Cn��                                    By���  �          AZ=q��=q�=qA�
Bz�Cw!H��=q�ə�A6=qBb�Cm&f                                    By��D  �          A[�
�����\A33B�Cv�������A3
=BZ{Clz�                                    By���  �          A[33��(����A  Bz�Cu���(���G�A3\)B[�Cl.                                    By���  �          AZ=q��
=���Ap�B�Cu����
=��33A0��BX�
Ck�f                                    By��6  �          AZ�R��  �A33Bp�CtJ=��  ��{A.�HBTp�Cj��                                    By���  �          AZ=q�����AffB�HCrٚ������HA-��BRCh�H                                    By��  T          AZ{���Q�A{B\)Cs@ ������A-G�BR�Ciu�                                    By��(  �          AZ=q��G����A�RBG�Ct  ��G���A.=qBS�HCj\)                                    By��  �          AW���G���\AG�BQ�Cs����G���=qA,  BS�\Ci�H                                    By�t  �          AW33���
=@�  B��Cq�H����ffA'
=BK��ChO\                                    By�$  �          AX����  ��RAG�B  Cu\��  ��  A/�BY
=CkQ�                                    By�2�  T          AX(���\)��RA��BCu&f��\)��Q�A/33BXCk}q                                    By�Af  �          AXz������A33B=qCvxR�����׮A.�RBW33CmxR                                    By�P  �          AY���=q��A�B\)Cw�
��=q��{A1�BZ�Cn�3                                    By�^�  �          AW������A�\B{CvL�����ָRA-��BVCmL�                                    By�mX  �          AW��\)��
A��B�Cx=q�\)����A-��BV�
Co��                                    By�{�  �          AX���w
=���A(�BCxz��w
=���HA2�HB^�\Co�H                                    By���  �          AW�
��Q��33A�B
=Ct� ��Q�����A0��B\  CjQ�                                    By��J  �          AW\)��p��z�A��B$��Cm����p���G�A1p�B^��C`E                                    By���  �          AW
=��{�=qA�\B&33Cr&f��{���
A4(�Bc�HCf)                                    By���  
�          AW���33��AffB%z�CqO\��33��33A4  Bbp�Ce)                                    By��<  �          AV�H�����{A�RB&�RCrT{�������A4Q�Bd\)CfT{                                    By���  �          AUp����
�p�A�B)33Cs�����
��=qA4��BgCg޸                                    By��  �          AU��w
=�{A
=B.{Cv� �w
=��G�A8Q�Bnz�Ck!H                                    By��.  T          AVff�U��AB1p�Cy�{�U���\A;\)Bs��Co
                                    By���  �          AVff�r�\�  A�\B2�Cv�{�r�\���A:�HBr�Cj�R                                    By�z  �          AW33�u�
�HA  B4
=Cv#��u����A;�
BsCi�                                    By�   �          AV�H�u��A�B/�
Cv�{�u��Q�A:{Bo�Ck�                                    By�+�  �          AX(��tz����A33B1�Cv�f�tz����RA;�
Bq�Ck                                    By�:l  �          AW�
�[��p�Ap�B/�CyO\�[���
=A;\)BqG�Co�                                    By�I  T          AX  �xQ���HA��B/p�Cv� �xQ���=qA:�RBo=qCk!H                                    By�W�  �          AX  ��G���A
=B=�\Cq����G����\A?\)By(�Cbٚ                                    By�f^  T          AW���  ��A z�B@ffCq�=��  ��p�A@  B{��Cb!H                                    By�u  �          AXz���{����A!�BA�RCpaH��{����A@��B{�\C_��                                    By���  �          AXz����H��
=A$��BFp�Cnp����H�z=qAAB~�C\u�                                    By��P  �          AXz����\��Q�A'�
BK��Ckn���\�Z�HAB=qB�
=CWQ�                                    By���  T          AW������ۅA(��BN�Ck!H�����P��AB�\B��CVT{                                    By���  �          AW\)�����(�A,Q�BT��Ck8R����?\)AD��B��)CT��                                    By��B  �          AW33���\����A'�
BM  Cj�����\�U�AAB�B�CV��                                    By���  �          AW33�����33A((�BMz�Cj�f����R�\AA��B�G�CV�                                    By�ێ  �          AV�\���\��ffA)�BO��Cj8R���\�H��AAB�(�CU                                      By��4  �          AV�\��(���z�A&�HBL
=Cj����(��W
=A@��B(�CV��                                    By���  �          AV�R���R���
A&�\BK\)Cj(����R�W
=A@(�B}��CV{                                    By��  �          AV{��(����A"=qBE(�Ci����(��g�A<��Bx  CW:�                                    By�&  �          AUp���
=���
AB>ChxR��
=�qG�A8��Bpz�CV��                                    By�$�  �          AT����{��p�A�B6  CfY���{�|��A3\)Bf�CU�{                                    By�3r  �          ATQ���p���Q�A�
B0p�Ce�H��p����A0(�B`�CU�f                                    By�B  �          AUG��ƸR���
A�B7Cc�H�ƸR�h��A3\)Be�CRaH                                    By�P�  T          AU���z���A�B=
=CcaH��z��X��A6{Bi\)CP޸                                    By�_d  �          AX����=q��Q�A$��BFffCZ���=q�Q�A7�
Bhp�CE^�                                    By�n
  �          AZ=q��ff��ffA(Q�BI�CXǮ��ff���
A9��Bi
=CBT{                                    By�|�  �          A[
=��
=���\A*=qBK�HCX���
=���A:�RBi��CA@                                     By��V  �          AZff��
=���A%p�BEp�CZ8R��
=��A8(�Bf��CD�                                    By���  �          AY��ff���
A$Q�BC�HCZ�R��ff�G�A7�Bf{CF!H                                    By���  �          AYp���=q���A%BF��C[8R��=q���A8��Bh��CE޸                                    By��H  �          AX����G�����A$��BEffC[Ǯ��G���
A8(�BhG�CF��                                    By���  �          AYG��ָR��33A&=qBG�RC[�H�ָR�\)A9p�Bjp�CFz�                                    By�Ԕ  �          AX���������HA%�BF�RC]�
�����   A9��Bk�RCH�                                    By��:  �          AXQ���{��z�A%G�BG=qC^n��{�"�\A9Bl��CI�=                                    By���  �          AX  ���
��
=A"ffBC  C`J=���
�:�HA8��Bj�CL�{                                    By� �  �          AXz��˅���A$��BF�C_�)�˅�.�RA:{Bm  CK5�                                    By�,  T          AX���˅��z�A&ffBHC^���˅�"�\A:�HBnz�CI�\                                    By��  
�          AX(���  ��ffA&{BH�
C]E��  �Q�A9��BlCH
                                    By�,x  �          AW�
�����A'�BKC\�
���{A:ffBn�HCG                                    By�;  �          AXz���\)��A(z�BLp�CY�{��\)����A9p�BkCCT{                                    By�I�  �          AW���  ��\)A)G�BN��CZ����  ���A:ffBoffCD.                                    By�Xj  �          AX����{���RA'33BIC[33��{�	��A9p�Bk�CE�=                                    By�g  �          AYp���G���p�A'\)BI(�CZ����G��
=A9p�Bi�HCEL�                                    By�u�  �          AZ�H������RA+
=BM�CW�������\)A:�RBjCA(�                                    By��\  
�          A[
=��ff���\A#�BA�RC\  ��ff�%A7�
Bd��CHn                                    By��  �          AZ�R���H��\)A'33BGffCXk����H���HA8Q�BeCCs3                                    By���  �          A[\)����{A'�BG
=CW�)����A8z�BdCB�3                                    By��N  �          A[
=������A((�BHz�CV���׿�33A7�Bc��C@�                                    By���  �          AZ�H��Q����A(��BI��CU���Q���A8  Bd��C?��                                    By�͚  �          AY���z���33A,  BPz�CSc���zῇ�A8��Bh  C<p�                                    By��@  �          AYG������  A+�BP�CP�3��녿:�HA6�RBe{C9�R                                    By���  �          AXQ����\(�A,��BT(�CM\���8Q�A5G�Bc��C5n                                    By���  �          AXz���\)�H��A-�BT��CJǮ��\)=�G�A4Q�Ba�RC30�                                    By�2  �          AX  ���H�333A,��BT�RCHB����H>�
=A2�\B^�C0�
                                    By��  �          AV�H��(��9��A*�\BRQ�CH����(�>�\)A0��B]p�C1��                                    By�%~  �          AW������HQ�A-�BU�CJ�f����=���A4(�Bb�C3B�                                    By�4$  �          AX����p��O\)A.=qBU�\CK�{��p�<#�
A5Bcp�C3�3                                    By�B�  �          AW���p��|(�A+\)BR�CP�\��p��:�HA6=qBfC9��                                    By�Qp  �          AU���
=��z�A!�BE�CRu���
=����A.ffB\��C>@                                     By�`  T          AS�
��(��N{A*=qBV�RCLQ���(���\)A1�BeQ�C4��                                    By�n�  �          ATQ���=q�K�A.�\B]��CL�q��=q=#�
A5Bl�C3��                                    By�}b  T          AUG���\)�FffA0��B`��CL�3��\)>��A7�BnffC2��                                    By��  �          AUp��ָR�W
=A0(�B_  CN�)�ָR��A8(�Bo�C5�                                    By���  �          AV�\�أ��h��A/�
B\
=CP=q�أ׾���A9�BnffC7c�                                    By��T  �          AV�H��  ����A0Q�B\CS���  �L��A;\)Bs�C:��                                    By���  �          AV�\�Å�g
=A5�Bh33CR�)�Å��\)A>�HB{z�C6��                                    By�Ơ  �          AU��G��p  A4��Bg\)CS�\��G���G�A>�\B|�C8#�                                    By��F  �          AUG������z�HA3�BeQ�CT�����ÿ!G�A=B{�C9��                                    By���  �          ATz������qG�A4Q�Bh�CT�\���;�A>{B~{C8�f                                    By��  �          ATQ������p�A4(�Bh{CX^�����aG�A?�B��HC<�\                                    By�8  �          AT����33��
=A-��BZffCY�q��33���
A;�BwG�CB�                                    By��  �          AU���=q��ffA/�
B^ffCX:���=q���RA<��By�C?�                                     By��  �          AT���������A/\)B]�
CWT{���Ϳ�A;�
Bw��C>                                    By�-*  �          AUp��������HA1�B`Q�CW��������\)A=G�Bz{C>�                                     By�;�  �          AU���p����\A/�B^{CW���p���33A<  Bwz�C>�                                    By�Jv  �          AT���ƸR��ffA,(�BXz�CY)�ƸR����A:=qBt�RCB.                                    By�Y  �          AS\)��p���
=A��BA�C\����p��3�
A0��Bc
=CJ�{                                    By�g�  �          AR�H��p���  A�\B<z�C]����p��H��A.�HB`  CM.                                    By�vh  �          AR=q�ə�����A#�BLC[���ə��33A4z�Bm  CH�                                    By��  �          AR{���H��ffA�RB=�
C^����H�FffA.�RBa=qCM(�                                    By���  �          AS\)�������AG�B@\)C]�q����AG�A1�Bcz�CL�R                                    By��Z  T          AS�������A�
BDQ�C\�����0  A2ffBe�HCJ��                                    By��   �          AP���θR���A\)B@=qC^5��θR�A�A.�HBcz�CM�                                    By���  �          APQ��Ϯ��p�A��B5��C`���Ϯ�l(�A*�RB\�CQ�
                                    By��L  �          AN�R������RAG�B?=qC_^�����J�HA-�Bc�CN��                                    By���  �          AN�H��ff���A"{BNCXT{��ff��{A0z�Bj�RCD\                                    By��  �          AN�H�ָR�}p�A$z�BSQ�CR� �ָR����A/\)BhC=�                                    By��>  �          AN�\��{��A.�RBg�CH�H��{?�A2�RBoC.�                                    By��  �          AN{��z��A1G�BmG�CH�f��z�?.{A4��Bu\)C-�f                                    By��  �          ANff��ff��RA0��Bl��CG��ff?G�A4(�BsC,��                                    By�&0  �          AO�
��Q��\)A4��Bq�\CHs3��Q�?Q�A7�Bx��C,33                                    By�4�  �          AO�
��\)���A0��Bh��CF����\)?J=qA3�Bo=qC-                                    By�C|  �          AP��������A2=qBjCF�3���?O\)A5�Bq=qC,�\                                    By�R"  �          AT  �=q?�  A$z�BM=qC+B��=q@z=qAp�B:��C\)                                    By�`�  �          AX���{@A&=qBH(�C$c��{@�\)A�\B/��CG�                                    By�on  �          AXz���@z�A'\)BJ��C&���@�\)A��B3��Cc�                                    By�~  �          AXz���?�A)G�BMC'ff��@���A�
B8
=C(�                                    By���  �          AX����R?���A*=qBO(�C'E��R@���A��B9\)C�f                                    By��`  �          A\  ��?��A-��BPz�C')��@�(�A�
B:z�C�H                                    By��  �          A]G���?�Q�A/33BQz�C&����@��RA!G�B;�C.                                    By���  �          A]����?��RA/
=BPG�C&�����@��A ��B9�C@                                     By��R  �          A^=q���?�
=A/\)BP\)C&�����@�{A!p�B:ffC�
                                    By���  �          A^ff�33?��A1�BS
=C'��33@�p�A#\)B=�CY�                                    By��  �          A]����?���A1��BT��C'����@�(�A$(�B>CB�                                    By��D  �          A]��\)?�A3
=BW=qC'0���\)@��A%��BAQ�C                                      By��  �          A^=q��ff?�z�A733B]�HC'ٚ��ff@���A*ffBH�C�)                                    By��  �          A^ff��{?�
=A7\)B^  C'���{@���A*�\BH(�C�                                     By�6  �          A_
=����?�\A6�HB\p�C'.����@��A)BF\)C\)                                    By�-�  �          A^�H��=q?�A6{B[  C&� ��=q@���A(��BDCB�                                    By�<�  �          A^�\���R?�\A7
=B]Q�C'
=���R@�33A*{BG=qC33                                    By�K(  �          A^ff����?�  A7�B^�C'#�����@�=qA*�RBH�C&f                                    By�Y�  �          A^�\��z�?�\A7�B^��C&����z�@��\A*�HBH��C
=                                    By�ht  �          A^=q���?�ffA733B^
=C&�����@�33A*=qBG��C                                    By�w  �          A^=q��33?�  A8  B_�C'\��33@���A+\)BI��C\                                    By�  �          A^�H����?�
=A6=qB[p�C&�����@�ffA(��BE{C޸                                    Byf  �          A_\)����?�33A6�RB[��C&T{����@��A)��BE�\C�                                    By£  �          A_���\)?��A7�B](�C&L���\)@��A*�\BG
=C�                                    By±�  �          A_����?���A8��B^C&�H���@�33A+�BH�HC                                      By��X  �          A_����?ٙ�A9�Ba=qC'Q����@�  A-��BK�
C:�                                    By���  �          A_33��R?�
=A;33Bd�C)#���R@�  A0  BP�CQ�                                    By�ݤ  �          A^{��\?���A:�RBe{C'���\@���A.�HBO�HC�                                    By��J  �          A]����Q�?�=qA;
=BfQ�C'����Q�@�(�A/\)BQ�C��                                    By���  �          A\����33?�p�A;
=Bg�
C&J=��33@�Q�A.�RBQz�C�\                                    By�	�  �          A[���G�@{A8��BeC"���G�@�ffA*�HBL��C��                                    By�<  �          AZ�R��Q�@*�HA6�RBb��C���Q�@�33A'
=BGG�C�f                                    By�&�  T          AZ�\��
=@C33A5p�B`��CQ���
=@�{A$z�BC33C�                                     By�5�  �          AY��
=@>�RA5G�B`�
C�)��
=@��A$��BC�C0�                                    By�D.  �          AY����z�?���A9G�Bi��C%5���z�@���A,��BR��C��                                    By�R�  �          AY����p�?���A;�
Bn��C&����p�@�33A0Q�BX��C�                                    By�az  �          AYp���p�?xQ�A?
=Bu��C+h���p�@q�A5Bc  C�=                                    By�p   �          AYp���z�?�A9p�Bj�C&\)��z�@��
A-BT\)C��                                    By�~�  �          AY���33?�
=A7
=Be��C&�f��33@��HA+�BP��C��                                    ByÍl  �          AZ=q�أ�?�z�A<��Bo�C*:��أ�@z=qA2�HB\=qC�                                    ByÜ  �          AZ{�Ӆ?B�\A>ffBs�C-u��Ӆ@a�A6=qBb��C�                                    Byê�  �          AY���Q�?&ffA?
=Bu{C.W
��Q�@Z�HA7\)Be33CL�                                    Byù^  �          AZ{��  ?+�A?\)Bu\)C.{��  @\(�A7�BeQ�C�                                    By��  �          AX�����
?�(�A<(�Bp�
C)����
@z�HA2�\B]��CY�                                    By�֪  �          AYG���  @�A7
=Be\)C"���  @���A)�BM�RC��                                    By��P  �          AYG�����@A8  Bg��C#�����@�{A+\)BPG�CǮ                                    By���  �          AZ=q��z�?�  A=�BpQ�C'L���z�@�A2�\B[��C��                                    By��  �          AZff�ə�?�(�A@��BwC)��ə�@|��A7
=Bc�C�3                                    By�B  �          A[
=��=q?�Q�A>ffBr�C'����=q@��
A4(�B]C��                                    By��  �          A[����
@�A;33Bi�
C#����
@�ffA.�RBR�RC�f                                    By�.�  �          A[
=��?�A=G�Bo=qC%���@�=qA2=qBY��C)                                    By�=4  �          AZ=q�Ϯ?�z�A>{Br\)C%���Ϯ@���A3
=B\Cp�                                    By�K�  �          AZ=q�У�@   A=�Bp{C"��У�@��
A0��BX�C��                                    By�Z�  �          AY���?�p�A=p�Bqz�C"޸��@��HA1p�BZ
=CxR                                    By�i&  �          AY���{?�\A=�Br��C$����{@�(�A2�RB\z�C�                                     By�w�  �          AY����?�A<��Bp�\C%�R���@�Q�A2{B[z�C�q                                    ByĆr  �          AX����ff@'�A9�BlffC����ff@�z�A,  BR
=Cp�                                    Byĕ  �          AYG����?�\)A>{Btp�C'�3���@~{A4z�B`��C=q                                    Byģ�  �          A[33��@%�A:�HBi��C�f��@�33A-�BP�C��                                    ByĲd  �          AZ�H��ff@EA:�\Bi��Ck���ff@��\A+33BM�RC�                                    By��
  �          AZff�θR@q�A6�\Bb��C�3�θR@�{A$��BD�C:�                                    By�ϰ  �          AW���  @��RA'
=BK{CT{��  @��
A��B(��C}q                                    By��V  �          AW\)����@�Q�A&�HBJ��C.����A=qA�B$�RB�8R                                    By���  �          AW\)��\)@�=qA&�RBK
=C�{��\)A
=AB$��B�B�                                    By���  �          AV�R��z�@���A'�BN�RC	޸��z�@�{AQ�B*{C ��                                    By�
H  �          AW��\@eA5��Bhp�Cp��\@��RA$��BI��C�
                                    By��  �          A��������@���A�G�CnO\����\@�\)B�HCi�=                                    By�'�  �          AQ��N{��@�  B&�\Cq� �N{��Q�@�(�BP��Cju�                                    By�6:  �          AG��N{��\)@�p�B$Q�Cq���N{���\@��BNffCj��                                    By�D�  T          A���C�
��ff@���B?�Co���C�
�j�H@�33Bgz�Cf(�                                    By�S�  �          A(��qG�����@�
=B>\)CfQ��qG��Dz�@���Ba
=C[(�                                    By�b,  �          A z���{�.�R@���B;�CR0���{��Q�@ǮBOffCE{                                    By�p�  �          @�\�}p�?^�R@ƸRBcG�C'���}p�@
=q@�(�BR{Cn                                    By�x  �          @�����?�ff@�33Bc  C%ff����@
=@��BPG�C��                                    ByŎ  �          @����tz�@��@�=qBYC�
�tz�@j�H@�B;�HC+�                                    ByŜ�  �          A33�\)@�33@�(�B7
=C:��\)@��@���B�B���                                    Byūj  �          AQ��5@�(�@�ffBPG�B�Ǯ�5@�@�z�B&��B�\                                    Byź  �          Aff����@��
@��
B%  C����@�\)@�A�  B�L�                                    By�ȶ  �          A=q�3�
@���@���B  B���3�
@�z�@`��A�p�B��                                    By��\  �          @�\)�H��@���@aG�A��B�{�H��@��
@��A�
=B�u�                                    By��  �          @陚��@�@<��A��B����@���?���AH(�B�33                                    By���  �          @��Ϳ�
=@ҏ\@<(�A��Bг3��
=@�G�?��A@��B��H                                    By�N  �          @��R��  @�@G�A�\)B�uÿ�  @�p�?�33AF=qB�#�                                    By��  �          @�\�G�@��H@#33A��B�z�G�@�R?���AB��)                                    By� �  �          @���
=@�p�?���AEB����
=@��H=���?E�B��=                                    By�/@  �          @߮���@ٙ�?�Q�A>ffB��)���@�ff=L��>�p�B��3                                    By�=�  �          @��=�G�@���?�@�ffB��=�G�@ȣ׿0�����B��                                    By�L�  �          @�����  @�\)?!G�@�
=B��R��  @���z����B��3                                    By�[2  �          @�
=?
=@��
=#�
>��B��f?
=@�  ���/\)B��3                                    By�i�  �          @Å?�  @���=p����B�Q�?�  @��H�����=qB�
=                                    By�x~  �          @�=q?��R@����G��p�B��?��R@��������
B�                                      ByƇ$  �          @�G�?O\)@���?E�A	B�Ǯ?O\)@�33�k��"�\B���                                    Byƕ�  �          @�{>��@�33�
=q����B�ff>��@�(���\)��B�                                    ByƤp  �          @�ff>��H@�z����  B�u�>��H@�p���33��33B��                                    ByƳ  �          @�Q�>\@Ǯ=�\)?&ffB�{>\@�zῑ��'�B���                                    By���  �          @�z�?333@�  �Tz����B�?333@����\��ffB�.                                    By��b  �          @�=q>�33@��þ�{��33B��R>�33@�(���  �q�B�p�                                    By��  �          @�=q>�ff@�  �&ff�߮B�#�>�ff@��ÿ�����B��R                                    By���  �          @�>��H@���8Q����B�W
>��H@���������B��                                    By��T  "          @�\)?��@��ÿz�H�z�B���?��@�\)�
=q��Q�B��)                                    By�
�  �          @�G�?���@�p����xQ�B�(�?���@����H�i�B���                                    By��  �          @���?s33@�G��k�����B��R?s33@�(�����H��B�W
                                    By�(F  �          @�z�?���@�\)�z����B��q?���@�  ��=q�v�HB���                                    By�6�  �          @�(�?333@��H�k�����B�?333@�p�����H  B��                                     By�E�  �          @ָR?�@��k���Q�B��?�@У׿��H�Ip�B��3                                    By�T8  �          @�?=p�@�(��W
=��ffB�33?=p�@�\)��
=�E�B��f                                    By�b�  �          @��?У�@�녿!G���\)B�
=?У�@ڏ\���t��B�.                                    By�q�  �          @�\)?�\)@�ff�+���  B���?�\)@�ff�G��s�B��H                                    Byǀ*  �          @��?�
=@�zᾊ=q��(�B�  ?�
=@�ff���H�E��B���                                    Byǎ�  �          A�\?k�Ap�������B��?k�@����G��G�B�Ǯ                                    Byǝv  �          A��?k�A�
����<��B�B�?k�A  ���T��B��                                    ByǬ  �          A�H>�A�\�\�
=B�  >�A
=�G��K\)B��)                                    ByǺ�  �          @��=#�
@�\)?@  @�(�B�\)=#�
@�  ��\��B�\)                                    By��h  �          @�G�?J=q@�\)��33�'�B�\)?J=q@�G���p��MG�B�\                                    By��  �          @�Q쿥�@�  ?�A��RB�B����@�\)?B�\@�33B�Q�                                    By��  �          @�Q��U@]p�@r�\B�HC�R�U@��\@G�A���CT{                                    By��Z  �          @����@dz�@�B��C�����@�G�@n�RA��
CL�                                    By�   �          @��H����@1G�@mp�B��C0�����@XQ�@J=qA�=qC�
                                    By��  �          A (���{@N�R@��Bp�C{��{@��@�p�B\)C�                                    By�!L  �          A���33@|(�@��B%�HCaH��33@��@��HB
Q�C��                                    By�/�  �          A	����p�@���@���B#�C���p�@�@�G�B�C�3                                    By�>�  �          A�����@�p�@��\BQ�C+�����@���@��A���B��                                    By�M>  �          Az����@��@��B��C^����@�33@��B �HB��
                                    By�[�  �          A�H����@��
@��\B33C 33����@��R@�(�A��HB�.                                    By�j�  �          A
�H�{�@�  @��RBB�{�{�@��H@�ffA�ffB�                                    By�y0  �          AQ��[�@���@��B�
B�W
�[�@��H@���A�
=B�B�                                    Byȇ�  �          A=q�q�@�ff@���B��B����q�@أ�@�=qA�
=B�L�                                    ByȖ|  �          A�
�w�@��H@��
B��B��f�w�@��@���A�\)B�G�                                    Byȥ"  �          A  �\��@���@��RB�HB���\��@�33@��RA��B���                                    Byȳ�  �          A����
@�  @�ffB	�B��{���
@ƸR@n{Aԣ�B��                                    By��n  �          A
{���@��R@�p�B\)C����@��@l��A��B�#�                                    By��  �          A�H����@���@��BffC z�����@ƸR@h��A�Q�B��{                                    By�ߺ  "          A  ���R@��R@��HB�RB��H���R@θR@��HAߙ�B�8R                                    By��`  �          A
ff��  @��@�Q�B�B����  @љ�@�  A�=qB��)                                    By��  �          A\)�vff@��@���BffB��R�vff@�ff@z=qA��
B�                                    By��  �          A=q�l(�@���@�  B=qB�B��l(�@�(�@���A�\)B�
=                                    By�R  �          A���Z�H@�p�@���B��B�.�Z�H@���@�G�A��B�=q                                    By�(�  �          A���e@���@���B��B�\�e@ٙ�@�  A�G�B�q                                    By�7�  �          A  �p  @�(�@��
B�B�
=�p  @ҏ\@w�A؏\B�aH                                    By�FD  �          A�
�o\)@�
=@�=qB{B��3�o\)@��@x��A��B�                                     By�T�  �          A��j=q@��\@�G�B%�HB�z��j=q@�z�@�p�B�
B�3                                    By�c�  �          AQ��u@�
=@���B!�
B����u@���@���B(�B�
=                                    By�r6  �          AQ���  @�ff@�B�
B�����  @�@���A���B��
                                    Byɀ�  �          A����G�@�\)@�B(�B�����G�@ƸR@���A�B�{                                    Byɏ�  �          A(��z�H@��@�Q�B  B����z�H@��@��
A�\)B��                                    Byɞ(  �          A��r�\@�ff@�z�B�\B���r�\@�@�G�A��HB�#�                                    Byɬ�  �          A�
�x��@���@���B��B��{�x��@�z�@�z�A�33B�                                    Byɻt  �          AQ��w
=@���@�33B�
B��q�w
=@�=q@�ffB��B���                                    By��  �          A\)�z=q@�(�@�  B��B��=�z=q@�=q@�33A�33B�z�                                    By���  �          A���xQ�@��@��B�B�ff�xQ�@�Q�@uA�z�B��                                    By��f  �          A�
��
=@�\)@�=qB\)B�G���
=@ƸR@�ffA��B�p�                                    By��  �          A\)���H@��@��B+{C ����H@�ff@�\)B�B�                                    By��  �          A  �y��@�ff@�=qB%��C B��y��@�
=@���B
�\B���                                    By�X  �          A�
���\@��R@�B"�C 
���\@�\)@�33B=qB���                                    By�!�  �          A���~�R@�p�@���Bp�B����~�R@�(�@�p�A���B�                                      By�0�  �          A���x��@���@�B33B����x��@�@�G�A�{B�ff                                    By�?J  �          A(��~{@�=q@�Q�B
=B����~{@�ff@w�A�ffB�L�                                    By�M�  �          A���}p�@��
@�Q�B�\B����}p�@�  @w�AׅB�q                                    By�\�  �          A=q�z=q@�@�ffB \)B��z=q@�Q�@c�
A��B���                                    By�k<  �          AQ�����@�
=@��BG�B��H����@��@j=qA˙�B�                                      By�y�  �          Az���{@�@��
A��B�L���{@أ�@l��A�\)B�                                    Byʈ�  �          A  ��Q�@��@�Q�BffB�33��Q�@�ff@��A�
=B��
                                    Byʗ.  �          A��z�@�p�@���B33B����z�@�p�@�z�A�B��R                                    Byʥ�  �          A�H��33@��\@�(�B��C G���33@У�@�Q�A��B�k�                                    Byʴz  �          A����
=@��@���B�B�B���
=@ҏ\@��A�(�B�ff                                    By��   �          Az���p�@�Q�@���B33B��R��p�@ָR@���A�z�B�Ǯ                                    By���  �          A���(�@�Q�@��B��B�����(�@�ff@���A�{B�3                                    By��l  �          Aff���\@أ�@�Q�B�\B�33���\@�R@�G�A��HB�k�                                    By��  �          A�����@ָR@���BQ�B�aH����@�@��A���B�                                    By���  �          A=q��ff@�(�@�=qB��B��
��ff@���@��HA��
B�Q�                                    By�^  �          A$Q��Mp�@�G�@���B3B�Q��Mp�@�R@��HB�HB��                                    By�  �          A*�R�Q�@߮@�33B.Q�B�B��Q�@���@�33B\)B�\                                    By�)�  �          A*�R�@��@�G�@�B0�\B�L��@��@��R@�{BffB�p�                                    By�8P  �          A+33�A�@�\)@���B+z�B�k��A�A{@ȣ�BffB��H                                    By�F�  �          A)���P��@�Q�@�{B  B���P��A�@���B=qB���                                    By�U�  �          A"ff�I��@�{@�B��B�G��I��A	G�@��A��B�W
                                    By�dB  �          A
=�g�@陚@�{A��
B����g�@���@N{A�ffB���                                    By�r�  �          AG��n�R@�ff@n�RA�ffB�3�n�RA@-p�A�
=B�aH                                    Byˁ�  �          A�����H@�z�@_\)A�\)B��)���HAQ�@��An=qB�3                                    Byː4  �          @�33�QG�@�
=@:�HA�p�B�\�QG�@�G�@z�AvffB�aH                                    By˞�  �          @�\)��{@  �   ��HB�uÿ�{?�p�����-33B��                                    By˭�  �          @��?����!G���Q��fC�8R?��������y��z�C��                                    By˼&  �          @S33?�{?�G���\)�(�Bf�
?�{?�=q��=q��B[33                                    By���  �          @ ��?8Q�?�  ���R�'=qB��3?8Q�?����z��@�Btff                                    By��r  �          ?��>�p�?
=�\(��MQ�Bg�>�p�>��n{�d�BQz�                                    By��  �          @y��>\)@N�R�\��p�B�#�>\)@AG���33� ��B���                                    By���  �          @����>{?���@�
=BcQ�C��>{?�z�@���BV  C5�                                    By�d  �          A{�o\)@?\)@�33B\�CaH�o\)@s33@�{BI�C��                                    By�
  �          A	��z�H@.{@��BaQ�C8R�z�H@c�
@���BP�C	�q                                    By�"�  �          A ���q�@C33@��BS�C��q�@r�\@�  BA�C޸                                    By�1V  �          @�ff�X��@dz�@�(�BH�C��X��@�  @�p�B3��C ��                                    By�?�  �          A ���c�
@e�@���BL��C޸�c�
@�G�@�=qB8C�                                    By�N�  �          A{��  @Q�@��HBOG�C�f��  @���@��B=(�C�\                                    By�]H  T          Az��s�
@u�@�z�BO��C���s�
@��@���B;C��                                    By�k�  �          A����{@dz�@�p�BQ��C� ��{@�(�@ָRB?�RC�3                                    By�z�  �          A���xQ�@Z=q@��B\�C
�3�xQ�@��@�
=BJ
=C}q                                    By̉:  �          A�����@dz�@��BM  C���@�33@ӅB;�CaH                                    By̗�  �          A{����@Tz�@��BI\)C8R����@��H@�33B9p�Ck�                                    By̦�  �          A�����@N{@��HBY�C:����@���@�BH�\C��                                    By̵,  �          A����G�@C33@��BU{C�=��G�@tz�@�p�BE
=C
G�                                    By���  �          A
�\�w
=@E�@�\B\Q�Cp��w
=@vff@�{BKQ�C�                                    By��x  �          A
=��  @;�@��B]
=C��  @mp�@�G�BL�HC	5�                                    By��  �          A	����\@7�@�
=BY�\C�{���\@g�@ӅBI�HC
h�                                    By���  �          A��|(�@N{@�=qBY  C�3�|(�@~{@�BH�C�q                                    By��j  �          A���vff@b�\@�(�BV�
C	ff�vff@�G�@�ffBE(�C�f                                    By�  T          A���S33@qG�@�33BbffC!H�S33@�=q@���BO
=B���                                    By��  �          A�H�j=q@^�R@��BbffC���j=q@���@�BP��C�)                                    By�*\  �          Az��Z�H@}p�@�{Bb�C޸�Z�H@���@�\)BO\)B�B�                                    By�9  �          A���L(�@k�A�Bmp�C���L(�@���@�G�BZffB�z�                                    By�G�  �          Aff�W
=@Mp�A�\Bq\)CG��W
=@���@���B_C�                                    By�VN  �          A��l(�@J=qA{Bl�CaH�l(�@\)@�Q�B[��C�q                                    By�d�  �          A�\�N�R@K�A�Bt�\C��N�R@�Q�@�33Bc  C �f                                    By�s�  �          A���G�@,��AG�B~��C&f�G�@b�\A (�Bn  Ck�                                    By͂@  �          AG��QG�@G�A�B��)CO\�QG�@5@�
=Bu�
C
�q                                    By͐�  �          A�G�?޸RA	B�=qC�
�G�@&ffA=qB�G�C(�                                    By͟�  �          A(���
=?.{A	G�B��C#׿�
=?��A�B�.B�8R                                    Byͮ2  �          A���Q�?��A�B�W
C���Q�@p�A=qB�z�C	�                                    Byͼ�  �          A���E�@#33A�B��CT{�E�@W
=A Q�Bq�Cu�                                    By��~  T          Ap��333?�z�A
�HB�L�C��333@0��A33B��CxR                                    By��$  �          A�Ϳ�=q?��
A��B���C����=q?�z�A�\B��HC޸                                    By���  �          A{�>{@�A��B�{C^��>{@L(�Az�By  C�3                                    By��p  �          A�W�@\)A33B���C^��W�@B�\A33Bt{C	��                                    By�  �          A\)�c�
?�Q�A��B���C��c�
@  A{B}=qCǮ                                    By��  �          AQ��g
=@&ffA{Bx=qC33�g
=@XQ�A��Bj\)C�{                                    By�#b  �          A=q�~�R@A33Bu�C�{�~�R@G�A33Bi\)C�H                                    By�2  �          Aff���H?�z�A{Br{C=q���H@+�A�RBh{C@                                     By�@�  �          A=q���H@ ��@�G�BW�RC�\���H@N{@���BMG�C��                                    By�OT  �          A����@
=@��
BH��C�����@1�@�z�B@ffCJ=                                    By�]�  �          AG���ff@ff@�\BIp�C�q��ff@0��@�BA=qC!H                                    By�l�  �          A���?�p�@�Q�BYC!H��@*�H@�G�BQG�C��                                    By�{F  �          A=q����?�(�AQ�Bx�\C#�����@.{A��Bn=qC�                                    ByΉ�  �          AQ��o\)?�z�AB�k�C��o\)@��A
�HB{  C�=                                    ByΘ�  �          Aff�\��?n{A33B�z�C$ٚ�\��?�(�AG�B��
Cp�                                    ByΧ8  �          A  �,(�?�\A�RB�L�C���,(�@"�\A�B�  C��                                    Byε�  �          A��j=q?z�HA	�B��{C%  �j=q?�p�A(�B��C�3                                    By�Ą  �          A\)�tz�?�A
=qB��C,(��tz�?��\A��B���C!�\                                    By��*  �          Ap��u��W
=AQ�B��qC7+��u�?
=qA  B�Q�C+�                                    By���  �          A���p  ��
=A��B�CL)�p  �p��A
�RB���CB
                                    By��v  �          A\)� ��?8Q�A��B��
C#�f� ��?��RA\)B��CT{                                    By��  �          A  �)��?�{A(�B�k�C�{�)��@A	�B�.C��                                    By��  �          Aff�
�H?�
=A=qB���C
h��
�H@*=qA33B��fC5�                                    By�h  �          A33��
@#33A��B�Q�C5���
@R�\A��B��fB�=q                                    By�+  �          A�1G�@G�A�B�.C���1G�@0  Az�B�\C5�                                    By�9�  �          A���C�
@ffA  B�aHC}q�C�
@C�
Az�B|ffC�                                    By�HZ  �          @��H��R?Ǯ@�z�B�\)C
=��R@Q�@�  B�u�CL�                                    By�W   �          @���{@{�@�A��
C�=��{@�33@   A��RCT{                                    By�e�  �          @������@��?��
A^�RCG����@��
?�A0��C}q                                    By�tL  �          A ����=q@�=q@|��A�p�CY���=q@��@e�A�p�C	��                                    Byς�  �          A�����H@��R@�{A�
=Cc����H@���@u�A�=qCz�                                    Byϑ�  �          @�z���Q�?˅@fffA�C&O\��Q�?�{@]p�Aأ�C$\                                    ByϠ>  �          @�
=���H@u�?���A)�Cٚ���H@{�?�=qAQ�C:�                                    ByϮ�  �          @��H����@\)>8Q�?���C!k�����@\)�#�
���
C!^�                                    ByϽ�  �          A��{@,(�?У�A@  C}q��{@3�
?�A'33C�                                    By��0  �          A�\���
@���@���A���CT{���
@��R@{�A���C�                                     By���  �          A=q���
@xQ�@���A�C�����
@��R@�  A�ffCz�                                    By��|  �          Ap�����@e�@��
B(�C�3����@{�@��HA��Ck�                                    By��"  T          @���
=>���@2�\A�
=C0O\��
=?��@0��A�Q�C.s3                                    By��  �          A z����H@��@qG�A�Q�C����H@�p�@Y��A��HC��                                    By�n  �          A ����  @�=q@h��Aי�CG���  @�=q@S33A�(�C
Ǯ                                    By�$  �          A����@���@^�RA�33C������@�Q�@I��A��RCz�                                    By�2�  �          A
=���@�
=@J�HA���C�����@�{@6ffA�G�CE                                    By�A`  �          A���=q@�
=@W�A�C����=q@�{@A�A�G�C��                                    By�P  �          A���=q@���@O\)A��C�{��=q@�\)@7�A�p�Cs3                                    By�^�  �          A������@�(�@UA�(�C�����@�33@?\)A��C
�                                    By�mR  �          A����33@�@a�Aə�CY���33@��@L��A�C�                                    By�{�  �          AG�����@�ff@��HA�\)C������@�
=@qG�A���C

=                                    ByЊ�  �          A\)����@��H@*�HA�33B�W
����@�Q�@�RAup�B��                                    ByЙD  �          A���x��@���@ffAd��B�G��x��@���?�{A.�HB�z�                                    ByЧ�  �          Az�����@�
=?���A,��B�G�����@��?�\)@�B��                                    Byж�  �          A(�����@أ�@Q�A�z�B��\����@��?�Q�AS
=B�u�                                    By��6  �          A������@�(�@J=qA�ffB�33����@�=q@/\)A�=qB���                                    By���  �          A	���  @�33@W�A�
=CQ���  @ə�@>{A�(�C ff                                    By��  �          A  ��{@θR@N{A��\B�Ǯ��{@��@333A�G�B�(�                                    By��(  �          Ap�����@�{@xQ�A�(�C�f����@�p�@`  A�ffC�\                                    By���  �          A
�R���@�  @k�AɅC�3���@�
=@S�
A�=qC��                                    By�t  �          A	����
=@��\@G�A�z�C
:���
=@�Q�@1�A�\)C	B�                                    By�  �          A	���(�@�G�@fffAŮC�\��(�@�  @Tz�A��CQ�                                    By�+�  �          A
{��Q�@��@XQ�A��RCY���Q�@��@G�A��
C&f                                    By�:f  �          A�����H@U@��A��HC�q���H@e@�ffA�ffCB�                                    By�I  �          AQ��Ϯ@Z=q@��B ��C@ �Ϯ@k�@���A�(�Cn                                    By�W�  �          A  ��
=@�z�@XQ�A�Q�C�f��
=@\@AG�A�Q�C                                      By�fX  T          A�����@���@j�HA�  C)���@�
=@UA�33C�                                    By�t�  �          A{����@�=q@s�
A��C����@���@^�RA�z�C�3                                    Byу�  �          A����H@��\@��HA�C�����H@��H@�Q�A�z�Cp�                                    ByђJ  �          A  ���@�
=@ǮB)\)B�����@��@�p�B�
B��\                                    ByѠ�  �          Ap���Q�@�p�@��
B#�
C�q��Q�@�  @���BC \                                    Byѯ�  �          A������@��H@�(�B	G�C	G�����@��
@��\B z�C                                    ByѾ<  �          A(���(�@��@1�A�33C)��(�@�ff@�RA�z�C
W
                                    By���  �          A�����@��
�p��~{Ck����@�
=�0����33C&f                                    By�ۈ  �          A���\)@2�\��z��<�C
=��\)@�H�����B=qC#�                                    By��.  �          A��  ��=q�����a�C7B���  �&ff��(��`Q�C;�
                                    By���  �          A	��c33@ ����z��tp�CxR�c33?�{��\)�y��C��                                    By�z  �          A
=q��?޸R���i�Cff��?�����{�m�HC"
                                    By�   �          AQ��Mp�@Q�����|{Cp��Mp�?�p�����C��                                    By�$�  �          A  ����@z���ff�]=qC����?�(�����b��C��                                    By�3l  �          A�
���@I�����
�1=qC�=���@7
=�����7p�CE                                    By�B  �          Aff��
=@B�\���R�=qC\��
=@1G����� z�C0�                                    By�P�  �          A{�y��@J=q��(��S�\C�R�y��@4z������Z�C�                                    By�_^  �          AQ��j=q?����(��s��C�{�j=q?�G���
=�xQ�C�\                                    By�n  �          A���{@<(������K(�Cs3��{@'�����Q�Cp�                                    By�|�  �          A�
��G�@n�R�Å�8�C�)��G�@[������?ffC�3                                    ByҋP  T          A�����@�=q����+��C
�����@s33��33�2��C
=                                    Byҙ�  �          A
=���
@_\)���
�9�Cs3���
@L�������@�CǮ                                    ByҨ�  �          A����Q�@mp����R�33Cz���Q�@^{��(�� z�CE                                    ByҷB  �          A���
=@����p��(�C{��
=@�33��  �&�\Cc�                                    By���  �          A����\@��?�{@�ffB������\@�ff?O\)@�=qB���                                    By�Ԏ  �          A�����@��?�p�A��B��f���@ָR?n{@�p�B��                                     By��4  �          A��q�@��?�Q�AZ�\B��q�@�33?�\)A6=qB�#�                                    By���  �          A�
��G�@�  ?�A<��B����G�@�=q?�\)A�\B�k�                                    By� �  �          A�R����@���?���AHz�B�k�����@�
=?\A'�B���                                    By�&  �          A(���Q�@��@ ��A��B�(���Q�@��@��Apz�B�u�                                    By��  �          A33��33@��?��@�33B���33@��?B�\@�  B�8R                                    By�,r  �          A�R��p�@�(�����{�B�aH��p�@�33�Y����Q�B�                                    By�;  �          A{�~{@��H��\)��B�u��~{@�\��p��(Q�B�                                    By�I�  �          A
=��ff@��H?   @b�\B�\��ff@ۅ>aG�?��
B��                                    By�Xd  �          A  �U@���@	��At(�B�{�U@�\)?�{AQ�B�\                                    By�g
  �          A��0��@�G�@��B&Q�B��H�0��@�Q�@�Q�B�B�#�                                    By�u�  �          A��7
=@��@��\B�\B����7
=@��@��HB=qB�u�                                    ByӄV  �          A��`  @��@��B��B�
=�`  @�@���B=qB�=q                                    ByӒ�  �          A�
�`  @��
@���B=qB���`  @�=q@�{B��B�8R                                    Byӡ�  �          A��j�H@�G�@��B��B����j�H@�\)@���B�B�8R                                    ByӰH  �          A��j�H@��@�33B(�B�G��j�H@��@�(�B��B�W
                                    ByӾ�  �          A
�R�n{@ə�@z=qA���B�
=�n{@�ff@j�HA��
B��f                                    By�͔  �          A(��Tz�A   ?�
=@�
=B�\�Tz�A ��?^�R@�\)B��)                                    By��:  �          A
=�[�@�(�>���@(�B�\�[�@�(�<�>k�B�
=                                    By���  �          A
ff�\(�@��Ϳ   �U�B���\(�@��
�J=q��Q�B�=q                                    By���  �          A���k�@�z�333��
=B�aH�k�@��z�H���HB�\                                    By�,  �          A{�{�@�׿���z�B��)�{�@�
=�����-p�B�33                                    By��  �          A\)�s�
@������W
=B��s�
@��H�Q��s�B�G�                                    By�%x  �          A����\@�z�����Q�B�B����\@ə��#33����B���                                    By�4  �          @���  @�����m�B�=q��  @��H�\)��B��f                                    By�B�  T          @��\����@�  ����S33C�q����@�{��(��iC{                                    By�Qj  �          @�p���@��H?�\@l��C ����@Å>���@(�C �                                    By�`  �          @�
=��
=@�?��@��HC �{��
=@�ff>Ǯ@5C �                                     By�n�  �          @��\��{@���>�@`��C�q��{@���>�\)@ffC�                                    By�}\  �          @�����@�33?�  @�C�����@�(�?Q�@�(�C��                                    ByԌ  �          @�\)���H@���?=p�@���C�����H@�G�?�@�Q�C�
                                    ByԚ�  �          @�����@�Q�G����C�
���@�\)�s33���C�q                                    ByԩN  �          @�����G�@�
=��z��(z�C����G�@�p���=q�<(�C�R                                    ByԷ�  �          @�  ���@�
=�E���  B�aH���@�ff�s33��{B���                                    By�ƚ  �          @���@  @�����?�B����@  @�녿�{�Yp�B��                                    By��@  �          @���dz�@ָR��{�?
=B�\�dz�@�����W�B�k�                                    By���  �          @����^{@�=q�h���أ�B���^{@�G���{���B�#�                                    By��  T          @�  ��  @�\)�{����B�����  @�p��������B�k�                                    By�2  �          @������@�(����\�(�B������@��H�����.�RB�G�                                    By��  �          @�\)�h��@�z���  �  B���h��@�Q���z��\)B�                                      By�~  �          @����K�@�ff������HB�8R�K�@��H��
=��B�=q                                    By�-$  �          @�  ��z�@��\��33�'=qB�z῔z�@�{��  �-�\B�{                                    By�;�  
�          @�  �e@�ff?��AH��B�8R�e@Ϯ?��HA2�RB��f                                    By�Jp  
�          @��R��z�@�\)?��AEG�B�.��z�@���?��HA0z�B���                                    By�Y  
�          @����@���?�
=A+�Bؽq��@�{?�p�A(�B؏\                                    By�g�  
�          @�  ���R@�R>��H@l(�Bʔ{���R@�
=>�z�@(�Bʏ\                                    By�vb  T          @��׿ٙ�@��H�Ǯ�<��B���ٙ�@�\�z����B�(�                                    ByՅ  �          @�\)��H@߮�޸R�TQ�B�B���H@�{��
=�j�RBڀ                                     ByՓ�  �          @��R��@����
=���HB�����@�33��\���B��                                    ByբT  T          @�{�8Q�@����A����B�G��8Q�@�ff�L(���{B��
                                    Byհ�  
Z          @�{�]p�@�������Q�B�.�]p�@�������
=B�u�                                    Byտ�  
�          @�ff�x��@����#�
��
=B����x��@�
=�-p����\B�8R                                    By��F  
�          @��H�z�H@�ff��G��XQ�B�=�z�H@����z��k
=B��                                    By���  �          @���z�@������,Q�C^���z�@�z´p��<(�C�\                                    By��  T          @��C�
@�
=�����'33B����C�
@�33��z��+�RB�L�                                    By��8  
�          @�  � ��@�G���33�7��B���� ��@����ff�<G�B�.                                    By��  
�          @�����@��\@<(�A�
=C�)����@�z�@5�A��CxR                                    By��  "          @�{��33@��H@��
B
=C
h���33@�@���B�\C	Ǯ                                    By�&*  
�          @����=q@�Q�@i��A癚C����=q@��H@c33A��\Cff                                    By�4�  S          @������R@�p�@mp�A�=qC�����R@�  @g�AݮC.                                    By�Cv  T          @�=q��z�@��@k�A��C
)��z�@�{@e�A��C	��                                    By�R  �          @�\)��{@�=q@���A�{C����{@��@z�HA�z�C{                                    By�`�  T          @�\)���@��@��RB�HC�\���@�  @�33B (�CO\                                    By�oh  
�          @������@�ff@�=qA�ffC���@�G�@~{A���C@                                     By�~  	�          @�G���p�@��R@r�\A���C=q��p�@�G�@l(�A�ffC�                                    By֌�  �          @�  ���
@�{@\(�A��HC	�=���
@�Q�@VffẠ�C	�                                    By֛Z  
�          @��H��Q�@�Q�@8Q�A�ffC0���Q�@�=q@1�A�ffC
ٚ                                    By֪   "          @����
=@��@"�\A�33CQ���
=@�G�@��A�\)C�                                    Byָ�  
Z          @��\��@��?�AZ{CW
��@���?�  AN�HC#�                                    By��L  
�          @�
=���R@�{@%A��C&f���R@�\)@ ��A��\Cٚ                                    By���  
�          @�
=��ff@��\@8��A��
CǮ��ff@�(�@3�
A��\Cp�                                    By��  
�          @�\)����@��@?\)A��
CE����@�
=@:=qA���C��                                    By��>  "          @�����@�Q�@4z�A��C���@��@/\)A�{Cn                                    By��  
�          @�G���\)@���@0��A�ffC8R��\)@�ff@,(�A���C��                                    By��  
�          @��R��z�@�p�@
=A�z�C!H��z�@��R@�\A���C޸                                    By�0  
�          @�=q����@��
?��A`��C�f����@���?�AVffC��                                    By�-�  "          @�����ff@�\)?�{A@(�C
�H��ff@�  ?\A5��C
�R                                    By�<|  �          @����@���?\A7
=CǮ���@��\?�Q�A-p�C��                                    By�K"  
�          @��R��  @�\)@��A��\C����  @���@�A�{C�{                                    By�Y�  
�          @����
@���?�
=AK�
CY����
@���?���AB�\C0�                                    By�hn  �          @�=q��=q@�z�?�\)Af�\C���=q@�p�?�ffA]p�C�q                                    By�w  "          @���p�@�
=?�Q�A1�C	����p�@��?�{A'33C	aH                                    Byׅ�  �          @�  ���@�z�>\@8��C0����@�z�>���@�\C(�                                    Byה`  �          @�
=��  @���=�\)?\)CQ���  @���    ���
CO\                                    Byף  "          @�������@��\����=qC������@�=q�&ff��ffC�3                                    Byױ�  
�          @����p�@��\������HCff��p�@�녿�33�p�Cz�                                    By��R  �          @�����\)@�
=?�p�A�\Cz���\)@��?�33A�CaH                                    By���  "          @����{@�(�?�Ak33C
���{@��?�Ab{C	�                                    By�ݞ  T          @�
=��G�@�ff?��A]C���G�@�\)?�(�AT��CǮ                                    By��D  
�          @�G���(�@�@��A�33C��(�@��R@Q�A���C
�
                                    By���  
�          @���\)@~�R@?\)A�G�C5���\)@���@;�A��C�                                    By�	�  �          @��
��p�@��@C�
A��
CY���p�@��@@��A�{C�                                    By�6  �          @������@|(�@W�A��C�f���@\)@S�
A�p�C�{                                    By�&�  
�          @�����@g�@c33A�=qC�H����@j�H@`  A���C�=                                    By�5�  �          @�p����@vff@��A�Q�CQ����@y��@��A���C�                                    By�D(  "          A\)����@n{@n�RA�z�CT{����@qG�@k�A�\)C                                      By�R�  T          @�p����@p��@i��A�  C�\���@s�
@fffAظRC:�                                    By�at  �          @��H���H@r�\@g
=A�G�C
=���H@u�@c�
A�{C�R                                    By�p  
Z          @�33��@s33@^�RA�(�CL���@vff@[�A���C                                      By�~�  T          @��H���
@s�
@b�\A���C���
@vff@`  A��
C�R                                    By؍f  �          @�33���
@j�H@Q�A�p�C����
@mp�@N�RA�z�C��                                    By؜  �          A����@qG�@_\)AˮC  ����@tz�@\(�A���C�R                                    Byت�  "          A(����H@u�@i��A�{C�����H@w�@g
=A�33C��                                    ByعX  T          A���\)@��H@|(�A�ffC���\)@�z�@xQ�A�33C�R                                    By���  �          AG�����@�{@z=qA߮CJ=����@��@w
=A܏\C                                      By�֤  
�          A�ȣ�@�Q�@s�
A؏\Ch��ȣ�@��@p��AծC!H                                    By��J  "          Az���z�@fff@uA��C�H��z�@h��@s33Aڏ\CW
                                    By���  �          AQ���Q�@�(�@w�A߮C���Q�@�p�@tz�AܸRC=q                                    By��  
�          A����H@�{@u�A�33C}q���H@�\)@q�A�=qC8R                                    By�<  �          A  ���@���@~{A�=qC����@��H@z�HA�p�C�f                                    By��  "          Az����R@|(�@�A�z�C�����R@~�R@�(�A�C@                                     By�.�  �          A  ����@r�\@�p�A�
=C�)����@u�@�(�A�ffC��                                    By�=.  �          A\)��\)@j�H@���A���Cs3��\)@n{@�\)A�{C!H                                    By�K�  �          AQ�����@l(�@�G�A�  C������@o\)@�  A��C5�                                    By�Zz  T          AG���G�@vff@�\)A���CxR��G�@y��@�{A�{C.                                    By�i   T          A��  @Y��@��B�Cc���  @\��@��RB�C                                    By�w�  T          A����@R�\@��HBG�C����@U@��B�C�{                                    Byنl  T          AQ���p�@]p�@���A�=qC����p�@`  @�Q�A�  Ch�                                    Byٕ  T          A=q��z�@`��@���A��C=q��z�@c33@\)A���C��                                    By٣�  �          A{���H@^{@��A��CQ����H@`��@��\A���C
=                                    Byٲ^  T          A{��(�@W�@�(�A�{C.��(�@Z=q@��HA��C�f                                    By��  "          A����Q�@O\)@�33B(�C����Q�@Q�@�=qB�C^�                                    By�Ϫ  "          A����@L��@�ffA��
C�����@O\)@�p�A��
C:�                                    By��P  �          A=q����@L��@�  A�{C������@O\)@�
=A�{C8R                                    By���  T          A��ȣ�@HQ�@���A�ffCxR�ȣ�@J�H@���A�z�C.                                    By���  	�          A���Q�@C�
@��
B ffC���Q�@Fff@�33A���C�f                                    By�
B  �          A�H��(�@E@�{Bz�CG���(�@HQ�@��B�\C��                                    By��  �          A�\����@C33@���B�RC������@E@�(�B��CQ�                                    By�'�  �          A���G�@E�@�=qA�G�C���G�@G
=@�G�A�p�C��                                    By�64  �          AQ��˅@Fff@���A��HC\�˅@HQ�@��A��CǮ                                    By�D�  �          A  ��{@C�
@���A��C�)��{@E@�(�A�(�CW
                                    By�S�  �          A  �Ӆ@=p�@|(�A��C�H�Ӆ@?\)@z�HA���C�H                                    By�b&  �          A���׮@AG�@s33A��
C�)�׮@C33@qG�A�=qC�H                                    By�p�  �          AG���
=@Dz�@w
=AܸRCp���
=@Fff@u�A�
=C5�                                    By�r  �          A����
=@E�@xQ�A�G�C\)��
=@G�@vffAۙ�C�                                    Byڎ  �          AG���33@G
=@eA��
C�
��33@H��@c�
A�=qCaH                                    Byڜ�  �          A{����@=p�@\��A¸RC33����@>�R@[�A�33C                                      Byګd  
�          A�
���@(��@g�AУ�C{���@*�H@fffA�33C޸                                    Byں
  �          AQ���{@0��@a�A�ffCJ=��{@2�\@`��A�
=C{                                    By�Ȱ  �          A�
�أ�@0  @tz�A���C���أ�@1�@r�\A�\)C��                                    By��V  �          Aff�׮@1G�@j=qA�33C�3�׮@333@hQ�A�Cz�                                    By���  �          A�R��p�@/\)@u�Aߙ�C�3��p�@1G�@s33A�(�CxR                                    By���  �          A�
�׮@&ff@}p�A��C��׮@(Q�@|(�A�\C��                                    By�H  �          A���H@%�@���A�{C\)���H@'
=@���A�RC�                                    By��  �          A���{@ff@|(�A�z�C!O\��{@Q�@{�A�G�C!{                                    By� �  �          A����\)@�\@w�A�ffC!�{��\)@z�@w
=A�33C!��                                    By�/:  �          A�����@p�@x��A�p�C aH���@\)@w�A�(�C &f                                    By�=�  �          A����{@��@x��A�33C!���{@�H@w�A�  C ��                                    By�L�  �          AQ���(�@�@x��A��\C ����(�@p�@w�A�\)C W
                                    By�[,  �          A����z�@
=@~{A�z�C!
��z�@��@|��A�G�C ٚ                                    By�i�  �          A�����@p�@|(�A�G�C �����@   @z�HA�  C�=                                    By�xx  �          A33����@(Q�@q�AۮC������@*=q@p��A�Q�C��                                    Byۇ  �          A�
����@)��@h��AхC������@+�@g
=A�(�C�                                     Byە�  �          A  ��=q@>�R@B�\A��HC&f��=q@@  @@��A�p�C��                                    Byۤj  �          A33��  @J=q@8Q�A�z�C�3��  @K�@6ffA���C�=                                    By۳  �          A��ᙚ@K�@2�\A���C���ᙚ@L��@0��A�G�C�{                                    By���  �          A
=��
=@Dz�@@��A�Q�C33��
=@Fff@>�RA��HC
=                                    By��\  �          A�޸R@Fff@1G�A�  C���޸R@G�@0  A�z�C�{                                    By��  �          A{��
=@?\)@:=qA�  CǮ��
=@AG�@8��A��\C�)                                    By���  �          Ap��޸R@@��@4z�A�G�C�{�޸R@B�\@333A�Ck�                                    By��N  �          Ap���p�@B�\@6ffA�33CL���p�@C�
@4z�A��C#�                                    By�
�  �          AG���ff@=p�@7�A�Q�C���ff@>�R@5A���C                                    By��  T          A{��G�@3�
@<(�A�\)C8R��G�@5@:=qA�  C�                                    By�(@  �          A���ff@*=q@)��A��\C����ff@,(�@(Q�A�33C�\                                    By�6�  �          A=q��p�@?\)@��A�
=CQ���p�@@��@�A��C0�                                    By�E�  �          A=q����@XQ�?��
AJ�HC������@X��?�  AG�C�)                                    By�T2  �          A���@;�@%�A�{C�R��@<��@#33A���C��                                    By�b�  T          A33��z�@6ff@6ffA��RC8R��z�@8Q�@5�A�G�C�                                    By�q~  �          A
=���@E�@7�A�=qCE���@G
=@6ffA��RC)                                    By܀$  T          A33����@Fff@7�A��C:�����@G�@5A�Q�C�                                    By܎�  �          A�
��\)@L(�@@��A�33Cp���\)@Mp�@>�RA��CE                                    Byܝp  �          A�
��(�@3�
@AG�A��C���(�@5�@?\)A�z�CW
                                    Byܬ  �          A�
��@6ff@B�\A�\)C!H��@8Q�@AG�A��
C�3                                    Byܺ�  �          A�
��@<��@<��A��Cs3��@>�R@:�HA�  CE                                    By��b  �          A���(�@<(�@6ffA�z�C�)��(�@=p�@5�A���Cp�                                    By��  �          A�H��\@<��@6ffA�
=CaH��\@>{@4z�A�p�C5�                                    By��  �          A33���@<(�@.{A��RC�3���@=p�@,(�A��C��                                    By��T  �          A���\)@:=q@(��A���C���\)@;�@'
=A�{C��                                    By��  �          A�
��@;�@(Q�A��C����@<��@'
=A��C��                                    By��  �          A  ��@<(�@*�HA���C�H��@>{@(��A�\)C�R                                    By�!F  �          A���ff@9��@0  A�  C
��ff@:�H@.{A�ffC��                                    By�/�  �          A���p�@8Q�@3�
A���C���p�@:=q@1�A�  C�                                    By�>�  
�          AQ���R@7
=@7�A�ffCff��R@8Q�@5A���C5�                                    By�M8  
�          AQ���\)@2�\@8��A��C�f��\)@4z�@7�A�Q�C��                                    By�[�  
�          AQ���
=@0��@;�A��C
��
=@2�\@9��A�ffC��                                    By�j�  
Z          A�
���@5�@<(�A�33Cu����@7
=@:=qA��CB�                                    By�y*  
�          A���@.{@=p�A��HCB���@0  @<(�A�G�C�                                    By݇�  �          A(�����@5@>�RA�\)CW
����@7�@<��A��C#�                                    Byݖv  	�          A���z�@?\)@C�
A��\CL���z�@AG�@A�A��RC
                                    Byݥ  
�          AG���@:=q@B�\A�\)C�R��@<(�@@��A��C                                    Byݳ�  �          A(����@1�@P  A�\)Cz����@4z�@N{A��C@                                     By��h  "          A�
��
=@0  @Z�HA�ffCu���
=@2�\@Y��A\C5�                                    By��  "          AQ���\@+�@UA�Q�CG���\@-p�@S�
A���C�                                    By�ߴ  
�          A(���p�@)��@H��A���C����p�@+�@G�A�33CxR                                    By��Z  S          AQ���z�@%@P��A��C ���z�@(Q�@N�RA�  C��                                    By��   �          A(����@.�R@S33A�Q�C�)���@1G�@QG�A�z�C��                                    By��  T          A(���\@.�R@P��A��C�3��\@0��@N�RA�{C��                                    By�L  T          A�
��z�@*=q@I��A���C����z�@,(�@G�A�CQ�                                    By�(�  "          A(���(�@+�@L��A�{Cc���(�@.{@J=qA�=qC!H                                    By�7�  
Z          AQ����@+�@I��A�\)Cz����@-p�@G�A��C8R                                    By�F>  �          A�
��(�@.{@G�A��C)��(�@0��@E�A�Cٚ                                    By�T�  
�          A\)��@8��@%A��HC@ ��@:�H@#33A���C
=                                    By�c�  �          A\)���@6ff@Dz�A�\)C����@8��@A�A�G�C                                    By�r0  �          A���@333@Dz�A��HC�=��@5�@A�A��HCG�                                    Byހ�  
(          A  ��(�@2�\@C�
A�Q�C�f��(�@5�@A�A�=qCc�                                    Byޏ|  T          A�
���
@1�@FffA���C����
@4z�@C�
A��\Ch�                                    Byޞ"  T          A����@5�@H��A��C.���@7�@FffA�\)C��                                    Byެ�  �          AQ����
@333@HQ�A�(�C�����
@5@FffA�  C@                                     By޻n  
�          A  ��\@4z�@I��A�CE��\@7
=@G
=A��C��                                    By��  T          A�����@0��@H��A�  C�f���@333@FffA�C��                                    By�غ  �          Az���(�@333@HQ�A��C�
��(�@5@FffA��CL�                                    By��`  "          A���\@:=q@P  A�(�C���\@<��@Mp�A��
CaH                                    By��  
�          A=q��G�@@��@Z�HA���C�\��G�@C�
@XQ�A�Q�C}q                                    By��  �          A33��\@C33@^{A��C�R��\@E@Z�HA��Cc�                                    By�R  �          A
=����@E@`  A��
CO\����@HQ�@\��A�33C��                                    By�!�  �          Aff�ᙚ@B�\@Z=qA��C���ᙚ@E@W
=A��HCT{                                    By�0�  
�          A����@AG�@`  A�
=C\)���@E�@\��A�Q�C�                                    By�?D  �          A  ���H@B�\@]p�AƏ\C����H@E@Z�HA�C�R                                    By�M�  #          A�H�ٙ�@@  @Z=qA�33C8R�ٙ�@C33@W�A�ffC޸                                    By�\�  
�          A�\��  @@��@^{A�\)C�3��  @C�
@[�A�z�C�{                                    By�k6  T          A�H��\)@C33@`��A�G�C�
��\)@G
=@]p�A�ffC5�                                    By�y�  �          A���Q�@E@c33A�ffCs3��Q�@H��@`  A�\)C\                                    By߈�  �          A\)�ڏ\@B�\@XQ�A�=qC  �ڏ\@Fff@U�A�G�C�H                                    Byߗ(  
�          A�R�ٙ�@A�@W�A���C�R�ٙ�@E@Tz�A��C�
                                    Byߥ�  
�          Aff��(�@<(�@N�RA���C�
��(�@?\)@K�A��CxR                                    Byߴt  �          A���ڏ\@8Q�@Q�A��C#��ڏ\@;�@N�RA�(�C                                    By��  T          A�
�ۅ@?\)@\(�A�p�Cu��ۅ@C33@X��A�ffC�                                    