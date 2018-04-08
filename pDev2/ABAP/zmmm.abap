REPORT zmm.
*____Class
*---------------------------------------------------------------------*
*   Method  test rest
*---------------------------------------------------------------------*
*   History :  1.03.2018 David Vela Tirado (EXTIG), Atos, FIRST_LIGHT
*              Creation of this method
*---------------------------------------------------------------------*
*____FM
*"----------------------------------------------------------------------
*"*"Local Interface:
*"  IMPORTING
*"     REFERENCE(PT_COMPONENTS) TYPE  /SYM/SC_ED_T_STPOA OPTIONAL
*"     REFERENCE(PF_MATNR) TYPE  MATNR
*"     REFERENCE(PF_STR) TYPE  CHAR01 DEFAULT 'X'
*"----------------------------------------------------------------------
 
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

    lv_url = 'hello/123' .
    lv_url = 'hello' .

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
        NEW-LINE.
        PERFORM process_answer USING json_response.
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

*************************************************************
****  TYPES
*************************************************************

  TYPES: BEGIN OF ty_json_req,
           hello TYPE string,
         END OF ty_json_req.

  TYPES: BEGIN OF ty_json_req2,
           world TYPE string,
         END OF ty_json_req2.

  DATA: json_req              TYPE ty_json_req,
        lr_json_deserializer  TYPE REF TO cl_trex_json_deserializer,
        lr_json_deserializer2 TYPE REF TO cl_trex_json_deserializer,
        tans                  TYPE string,
        json_req2             TYPE ty_json_req2..

*************************************************************
****  PROCESS JSON
*************************************************************
FORM process_answer USING ans.
*  ans= {"hello": "{'world':'10'}"}  convert to abap structure and get values!
*

*  REPLACE ALL OCCURRENCES OF '#' IN ans WITH ''. " not a problem!
  REPLACE ALL OCCURRENCES OF '"hello"' IN ans WITH 'hello'.
  WRITE ans.
  NEW-LINE.

  TRY.
      tans = '{hello:"hola"}'. " right format!

*      tans = '{hello:"' && '{' && |'world':'1'| && '}'&&   '"}'. " a lot of work


*      tans = '{' && |hello:'hola'| && '}'.  " replace ' for "!!!
*      REPLACE ALL OCCURRENCES OF |'| IN tans WITH '"'.
      WRITE tans. NEW-LINE.
      CREATE OBJECT lr_json_deserializer.
      lr_json_deserializer->deserialize( EXPORTING json = tans IMPORTING abap = json_req ).
    CATCH cx_root INTO DATA(lo_ex).
      WRITE 'exception!'.
  ENDTRY.
  WRITE json_req-hello.
  RETURN.

  WRITE json_req2-world.
  REPLACE ALL OCCURRENCES OF |'world'| IN json_req2-world WITH 'world'.
  REPLACE ALL OCCURRENCES OF |'| IN json_req2-world WITH '"'.

  CREATE OBJECT lr_json_deserializer2.
  lr_json_deserializer2->deserialize( EXPORTING json = ans IMPORTING abap = json_req2 ).
  write json_req2-world.

ENDFORM.


*  json_req-salesorder = '25252525'.
*  json_req-type = 'Direct'.

*  DATA lr_json_serializer   TYPE REF TO cl_trex_json_serializer.
*  CREATE OBJECT lr_json_serializer EXPORTING data = json_req.
*
*  lr_json_serializer->serialize( ).
*  lv_body = lr_json_serializer->get_data( ).

* Or you can use this class as well
* lv_body = /ui2/cl_json=>serialize( data = json_req ).
* Converted JSON should look like this
* lv_body = '{ "salesorder":"25252525", "type":"Direct"}'.