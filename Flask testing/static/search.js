<script>
//GOOGLE SEARCH
//Enter domain of site to search
var domainroot="example.com"
function Gsitesearch(curobj){
curobj.q.value="site:"+domainroot+" "+curobj.qfront.value
}
</script>
<form class="search" action="http://www.google.com/search" method="get" onSubmit="Gsitesearch(this)">
<input name="q" type="hidden" />
<input name="qfront" type="search" required class="searchField" placeholder="Google Site Search" maxlength="50"/>
<input type="submit" class="search-button" value="SEARCH" />
</form>