$("#model").change(function(){
    if($(this).val() == "ElasticNet()"){
      $("#alpha").show();
    }else{
      $("#alpha").hide();
    }

});
$("#model2").change(function(){
    if($(this).val() == "ElasticNet()"){
      $("#alpha2").show();
    }else{
      $("#alpha2").hide();
    }

});