
REPORT zmm.

PARAMETERS p_txt TYPE string DEFAULT 'WORLD'.

START-OF-SELECTION.

  WRITE 'Hello  ' && p_txt.
  NEW-LINE.
*  RETURN.

  DATA: lo_http_client TYPE REF TO if_http_client,
        lo_rest_client TYPE REF TO cl_rest_http_client,
        lo_response    TYPE REF TO     if_rest_entity,
        lv_url         TYPE string,
        lv_body        TYPE string,
        token          TYPE string,
        agreements     TYPE string,
        data           TYPE string.
        "Create a structure(or deep) that exactly matches your JSON response
*       abap_response  TYPE        zca_serno_ln,



* Create HTTP intance using RFC restination created
* You can directly use the REST service URL as well
  cl_http_client=>create_by_destination(
   EXPORTING
     destination              = 'FIRSTLIGHT'    " Logical destination (specified in function call)
   IMPORTING
     client                   = lo_http_client    " HTTP Client Abstraction
   EXCEPTIONS
     argument_not_found       = 1
     destination_not_found    = 2
     destination_no_authority = 3
     plugin_not_active        = 4
     internal_error           = 5
     OTHERS                   = 6
 ).



* If you are using cl_http_client=>create_by_url use this code to supress and pass your
* HTTP basic authenication
*  lo_http_client->propertytype_logon_popup = lo_http_client->co_disabled.
*  DATA l_username TYPE string.
*  DATA l_password TYPE string.
*  l_username = 'user'.
*  l_password = 'password'.
*  CALL METHOD lo_http_client->authenticate
*    EXPORTING
*      username = l_username
*      password = l_password.

*  lo_http_client
* Create REST client instance
  CREATE OBJECT lo_rest_client
    EXPORTING
      io_http_client = lo_http_client.

* Set HTTP version
  lo_http_client->request->set_version( if_http_request=>co_protocol_version_1_0 ).
  IF lo_http_client IS BOUND AND lo_rest_client IS BOUND.
*    DATA(id) = 'XYZ'.
*    CONCATENATE '/agreements/' id INTO lv_url.

* curl http://localhost:5000/F1L -d "data='{"m":"1","100023":1}'" -X GET
    data = '{"m":"1","100023":1}'.
    data = '-d "data" = ''' &&  '{"m":"1","100023":1}'''.
*    data = ' -d "data"=''' &&  '{"m":"1","100023":1}'''.

    lv_url = '/123' .

    TRY.
* Set the URI if any
        cl_http_utility=>set_request_uri(
          EXPORTING
            request = lo_http_client->request    " HTTP Framework (iHTTP) HTTP Request
            uri     = lv_url                     " URI String (in the Form of /path?query-string)
        ).

*        cl_http_utility=>set_query(
*          EXPORTING
*            request = lo_http_client->request      " HTTP Framework (iHTTP) HTTP Request
*            query   = data   " Query String (in Form of form_field1=1&form_field2=2)
*        ).
*

*        lo_rest_client->if_rest_client~post( io_entity =  ).


* Set request header if any
*     CALL METHOD lo_rest_client->if_rest_client~set_request_header
*       EXPORTING
*         iv_name  = 'auth-token'
*         iv_value = token.

* HTTP GET
        lo_rest_client->if_rest_client~get( ).
* HTTP response
        lo_response = lo_rest_client->if_rest_client~get_response_entity( ).
* HTTP return status
        DATA(http_status)   = lo_response->get_header_field( '~status_code' ).
* HTTP JSON return string
        DATA(json_response) = lo_response->get_string_data( ).
        WRITE json_response.
        RETURN.
* Class to convert the JSON to an ABAP sttructure
*    DATA lr_json_deserializer TYPE REF TO cl_trex_json_deserializer.
*    CREATE OBJECT lr_json_deserializer.
*    lr_json_deserializer->deserialize( EXPORTING json = json_response IMPORTING abap = abap_response ).

      CATCH cx_rest_client_exception.
        WRITE 'exception - REST WS unavailable'.
      CATCH cx_root.
        WRITE 'exception cx_root'.
    ENDTRY.

  ELSE.
    WRITE 'no RFC found'.
  ENDIF.